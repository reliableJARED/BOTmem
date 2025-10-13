import subprocess
import sys
import json
import re
import os
import socket
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime, timezone, timedelta
import torch
import numpy as np

def install_dependencies():
    """Install required packages if not available."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import spacy
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "spacy"])
        subprocess.check_call([sys.executable, "-m", "python", "-m", "spacy", "download", "en_core_web_sm"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import spacy
    
    return torch, AutoModelForCausalLM, AutoTokenizer

def check_internet():
    """Check if internet connection is available."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except (socket.timeout, socket.error, OSError):
        return False

def find_local_model(model_name):
    """Find cached model in common HuggingFace cache locations."""
    local_name = model_name.split('/')[-1]
    if os.path.exists(local_name) and validate_model_files(local_name):
        return local_name
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)
    
    if os.path.exists(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            for snapshot in snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if validate_model_files(snapshot_path):
                    return snapshot_path
    
    return None

def validate_model_files(model_path):
    """Check if model directory has required files."""
    if not os.path.exists(model_path):
        return False
    
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    
    return len(model_files) > 0

def load_model(model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
    """Load model and tokenizer with offline support."""
    torch, AutoModelForCausalLM, AutoTokenizer = install_dependencies()
    
    if force_offline or not check_internet():
        print("Using offline mode...")
        local_path = find_local_model(model_name)
        if not local_path:
            raise FileNotFoundError(
                f"Model {model_name} not found locally. "
                f"Please run with internet connection first to download the model."
            )
        
        print(f"Loading model from: {local_path}")
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True
        )
    else:
        print(f"Loading {model_name} (will download if needed)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer, torch


class BoTMemory:
    """Temporal Bag of Transforms Memory with LLM-Generated Context."""
    
    def __init__(self, model, tokenizer, torch_module):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.device = model.device
        
        # Memory database with temporal support
        self.memory_db = {}
        
        # Load spaCy for NLP processing
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Installing spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        
        # Get embedding dimension
        self.embedding_dim = model.get_input_embeddings().weight.shape[1]
        
        # Cache for transforms at different times
        self.transform_cache = {}
    
    def _parse_attributes(self, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Parse string attributes into weighted dictionaries using LLM.
        """
        if not attributes:
            return []
        
        # Ask LLM to weight the attributes
        attr_str = ", ".join(attributes)
        prompt = f"""Analyze these attributes: {attr_str}

Rate each attribute's importance for defining an entity (0.0 to 1.0):
- Entity types (cat, dog, person, place) should be 0.9-1.0
- Key characteristics (personality, age) should be 0.6-0.8  
- Physical descriptions (colors, appearance) should be 0.4-0.6
- Minor details should be 0.2-0.4

Return ONLY a JSON object with attribute:weight pairs.
Example: {{"cat": 1.0, "playful": 0.7, "black": 0.5}}"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            with self.torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                weights_dict = json.loads(json_match.group(0))
            else:
                # Fallback to uniform weights
                weights_dict = {attr.lower().replace("_", " "): 0.5 for attr in attributes}
        
        except Exception as e:
            print(f"  Warning: LLM weighting failed ({e}), using defaults")
            weights_dict = {attr.lower().replace("_", " "): 0.5 for attr in attributes}
        
        # Convert to list format
        weighted = []
        for attr in attributes:
            attr_clean = attr.lower().replace("_", " ")
            weight = weights_dict.get(attr_clean, 0.5)
            weighted.append({
                "name": attr_clean,
                "weight": float(weight),
                "active_from": datetime.now(timezone.utc),
                "active_until": None
            })
        
        return weighted
    
    def _get_active_attributes(self, entity_name: str, at_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get attributes active at a specific time."""
        if entity_name not in self.memory_db:
            return []
        
        if at_time is None:
            at_time = datetime.now(timezone.utc)
        
        active = []
        for attr in self.memory_db[entity_name]["attributes"]:
            active_from = attr["active_from"]
            active_until = attr["active_until"]
            
            if active_from <= at_time and (active_until is None or at_time < active_until):
                active.append({
                    "name": attr["name"],
                    "weight": attr["weight"]
                })
        
        return active
    
    def _generate_context_description(self, entity_name: str, attributes: List[Dict[str, Any]]) -> str:
        """
        Use LLM to generate a rich contextual description that preserves the entity name.
        """
        if not attributes:
            return entity_name
        
        # Sort by weight
        sorted_attrs = sorted(attributes, key=lambda x: x["weight"], reverse=True)
        
        # Build attribute string emphasizing important ones
        important = [a["name"] for a in sorted_attrs if a["weight"] >= 0.8]
        moderate = [a["name"] for a in sorted_attrs if 0.5 <= a["weight"] < 0.8]
        minor = [a["name"] for a in sorted_attrs if a["weight"] < 0.5]
        
        # Create prompt that preserves entity name
        prompt = f"Complete this description of {entity_name}:\n\n"
        
        if important:
            prompt += f"{entity_name} is a {important[0]}"
            if len(important) > 1:
                prompt += f" ({', '.join(important[1:])})"
            if moderate:
                prompt += f" who is {', '.join(moderate)}"
            prompt += f". {entity_name}"
        else:
            attrs_str = ", ".join([a["name"] for a in sorted_attrs[:4]])
            prompt += f"{entity_name} has these characteristics: {attrs_str}. {entity_name}"
        
        # Ask for completion
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Combine prompt start and completion
        full_description = prompt.split(f". {entity_name}")[-1] + " " + completion
        
        # Ensure entity name is preserved
        if entity_name not in full_description:
            full_description = f"{entity_name} " + full_description
        
        # Clean up
        if '\n' in full_description:
            full_description = full_description.split('\n')[0]
        
        return full_description
    
    def _generate_behavior_description(self, entity_name: str, entity_type: str) -> str:
        """
        Use LLM to generate typical behaviors for the entity type.
        """
        prompt = f"List 4 typical behaviors or actions that a {entity_type} named {entity_name} would do:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        behaviors = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean up and extract key words
        if '\n' in behaviors:
            behaviors = behaviors.split('\n')[0]
        
        # Keep it concise
        behavior_words = behaviors.split()[:8]  # Limit to 8 words
        return " ".join(behavior_words)
    
    def _compute_bag_of_transforms(self, entity_name: str, attributes: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Compute multiple transform vectors preserving entity name.
        """
        if not attributes:
            return {}
        
        # Generate contextual description that includes entity name
        description = self._generate_context_description(entity_name, attributes)
        print(f"  → Context: '{description[:80]}...'")
        
        transforms = {}
        
        # 1. Compute entity transform (preserves name)
        desc_tokens = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        
        with self.torch.no_grad():
            embedding_layer = self.model.get_input_embeddings()
            
            # Description embedding (includes entity name)
            desc_emb = embedding_layer(desc_tokens['input_ids'])
            attention_mask = desc_tokens['attention_mask'].unsqueeze(-1)
            masked_emb = desc_emb * attention_mask
            sum_emb = masked_emb.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            desc_vector = sum_emb / sum_mask.clamp(min=1)
            desc_vector = desc_vector.squeeze(0)
            
            transforms["entity"] = desc_vector
            
            # 2. Find primary type for behavior generation
            primary_type = None
            for attr in attributes:
                if attr["weight"] >= 0.9:
                    primary_type = attr["name"]
                    break
            
            if primary_type:
                # Get LLM-generated behaviors for this type
                behaviors = self._generate_behavior_description(entity_name, primary_type)
                
                if behaviors:
                    behavior_tokens = self.tokenizer(behaviors, return_tensors="pt", padding=True).to(self.device)
                    behavior_emb = embedding_layer(behavior_tokens['input_ids'])
                    behavior_emb = behavior_emb.mean(dim=1).squeeze(0)
                    transforms["behavior"] = behavior_emb
                    print(f"  → Behaviors: '{behaviors[:50]}...'")
        
        return transforms
    
    def add_entity(self, entity_name: str, attributes: List[str], at_time: Optional[datetime] = None):
        """
        Add an entity with temporal attributes.
        
        Args:
            entity_name: Name of the entity (e.g., "Mickey")
            attributes: List of attribute strings
            at_time: When this entity definition becomes active
        """
        if at_time is None:
            at_time = datetime.now(timezone.utc)
        
        # Parse attributes with LLM weighting
        print(f"\nAdding entity '{entity_name}' (active from {at_time.strftime('%Y-%m-%d')})")
        print(f"  Analyzing attributes...")
        
        weighted_attrs = self._parse_attributes(attributes)
        for attr in weighted_attrs:
            attr["active_from"] = at_time
        
        print(f"  Weighted attributes:")
        for attr in weighted_attrs:
            print(f"    - {attr['name']}: {attr['weight']:.2f}")
        
        # Initialize or update entity
        if entity_name not in self.memory_db:
            self.memory_db[entity_name] = {
                "attributes": [],
                "transforms": {}
            }
        
        # Add new attributes
        self.memory_db[entity_name]["attributes"].extend(weighted_attrs)
        
        # Compute transforms for this time point
        self._recompute_transforms(entity_name, at_time)
    
    def update_entity(self, entity_name: str, attributes: List[str], at_time: Optional[datetime] = None):
        """
        Update entity (deactivate old attributes, add new ones).
        """
        if at_time is None:
            at_time = datetime.now(timezone.utc)
        
        if entity_name not in self.memory_db:
            self.add_entity(entity_name, attributes, at_time)
            return
        
        # Deactivate all currently active attributes
        for attr in self.memory_db[entity_name]["attributes"]:
            if attr["active_until"] is None:
                attr["active_until"] = at_time
        
        # Add new attributes with LLM weighting
        print(f"\nUpdating entity '{entity_name}' (as of {at_time.strftime('%Y-%m-%d')})")
        print(f"  Analyzing new attributes...")
        
        weighted_attrs = self._parse_attributes(attributes)
        for attr in weighted_attrs:
            attr["active_from"] = at_time
        
        print(f"  New weighted attributes:")
        for attr in weighted_attrs:
            print(f"    - {attr['name']}: {attr['weight']:.2f}")
        
        self.memory_db[entity_name]["attributes"].extend(weighted_attrs)
        
        # Recompute transforms
        self._recompute_transforms(entity_name, at_time)
    
    def _recompute_transforms(self, entity_name: str, at_time: datetime):
        """Compute and cache transforms for a specific time."""
        active_attrs = self._get_active_attributes(entity_name, at_time)
        
        if not active_attrs:
            print(f"  ⚠ No active attributes at {at_time}")
            return
        
        # Compute bag of transforms
        transforms = self._compute_bag_of_transforms(entity_name, active_attrs)
        
        # Cache with timestamp
        time_key = at_time.strftime("%Y-%m-%d")
        self.memory_db[entity_name]["transforms"][time_key] = transforms
        
        # Calculate importance
        importance = max((a["weight"] for a in active_attrs), default=0.5)
        print(f"  ✓ Transforms computed (importance: {importance:.2f})")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7,
                 at_time: Optional[datetime] = None) -> str:
        """
        Generate text with BoT memory applied at specific time.
        """
        if at_time is None:
            at_time = datetime.now(timezone.utc)
        
        # Check for entities in prompt
        entities_found = self._find_entities_in_text(prompt)
        
        if entities_found:
            # Inject context for found entities
            modified_prompt = self._inject_context(prompt, entities_found, at_time)
            
            # Tokenize modified prompt
            model_inputs = self.tokenizer(modified_prompt, return_tensors="pt", padding=True).to(self.device)
        else:
            # No entities found, use original prompt
            model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        
        # Get embeddings and apply transforms
        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids)
        
        # Apply entity transforms with amplification
        if entities_found:
            embeddings = self._apply_transforms(embeddings, modified_prompt if entities_found else prompt, 
                                              entities_found, at_time)
        
        # Generate with modified embeddings
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode output
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Remove injected context from output if present
        if entities_found and "[Context:" in output_text:
            # Extract just the generated part after context
            parts = output_text.split("]")
            if len(parts) > 1:
                # Get everything after context marker
                clean_output = parts[-1].strip()
                # Restore original prompt
                if prompt in clean_output:
                    output_text = clean_output
                else:
                    output_text = prompt + " " + clean_output.replace(modified_prompt.split("]")[-1].strip(), "").strip()
        
        return output_text
    
    def _find_entities_in_text(self, text: str) -> Dict[str, tuple]:
        """Find entities in text."""
        entities_found = {}
        text_lower = text.lower()
        
        for entity_name in self.memory_db.keys():
            entity_lower = entity_name.lower()
            if entity_lower in text_lower:
                pos = text_lower.find(entity_lower)
                entities_found[entity_name] = (pos, pos + len(entity_name))
        
        return entities_found
    
    def _inject_context(self, prompt: str, entities_found: Dict[str, tuple], at_time: datetime) -> str:
        """
        Inject context that preserves entity names.
        """
        context_parts = []
        
        for entity_name in entities_found.keys():
            # Get active attributes
            active_attrs = self._get_active_attributes(entity_name, at_time)
            
            if active_attrs:
                # Find primary type
                primary_type = None
                for attr in active_attrs:
                    if attr["weight"] >= 0.9:
                        primary_type = attr["name"]
                        break
                
                if primary_type:
                    # Include entity name in context
                    context_parts.append(f"{entity_name} is a {primary_type}")
        
        if context_parts:
            # Inject context at the beginning
            context = "[Context: " + ". ".join(context_parts) + ".] "
            return context + prompt
        
        return prompt
    
    def _apply_transforms(self, embeddings: torch.Tensor, text: str, entities_found: Dict[str, tuple],
                         at_time: datetime) -> torch.Tensor:
        """
        Apply transforms with strong replacement.
        """
        # Get offset mapping for the text
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoding['offset_mapping']
        
        for entity_name, (start_pos, end_pos) in entities_found.items():
            # Get transforms for this time
            time_key = at_time.strftime("%Y-%m-%d")
            
            # Compute if not cached
            if time_key not in self.memory_db[entity_name]["transforms"]:
                self._recompute_transforms(entity_name, at_time)
            
            if time_key not in self.memory_db[entity_name]["transforms"]:
                continue
            
            transforms = self.memory_db[entity_name]["transforms"][time_key]
            
            if "entity" not in transforms:
                continue
            
            entity_transform = transforms["entity"]
            
            # Find token positions for entity
            entity_tokens = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end:  # Skip special tokens
                    continue
                if token_start < end_pos and token_end > start_pos:
                    entity_tokens.append(token_idx)
            
            # Apply very strong transform (0.9 replacement)
            for token_idx in entity_tokens:
                if token_idx < embeddings.shape[1]:
                    base_emb = embeddings[0, token_idx].clone()
                    # Almost complete replacement while preserving some original signal
                    embeddings[0, token_idx] = 0.1 * base_emb + 0.9 * entity_transform
            
            # Apply behavior influence if available
            if "behavior" in transforms and entity_tokens:
                behavior_transform = transforms["behavior"]
                # Apply to tokens immediately after entity
                last_entity_token = max(entity_tokens)
                for i in range(1, min(3, embeddings.shape[1] - last_entity_token)):
                    token_idx = last_entity_token + i
                    base_emb = embeddings[0, token_idx].clone()
                    # Moderate influence on nearby tokens
                    embeddings[0, token_idx] = 0.6 * base_emb + 0.4 * behavior_transform
        
        return embeddings


class BoTMemorySimple:
    """Simple interface for BoT Memory."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
        self.model, self.tokenizer, self.torch = load_model(model_name, force_offline)
        self.bot = BoTMemory(self.model, self.tokenizer, self.torch)
    
    def add_entity(self, entity_name: str, attributes: List[str], at_time: Optional[datetime] = None):
        """Add entity with list of attributes."""
        self.bot.add_entity(entity_name, attributes, at_time)
    
    def update_entity(self, entity_name: str, attributes: List[str], at_time: Optional[datetime] = None):
        """Update entity attributes."""
        self.bot.update_entity(entity_name, attributes, at_time)
    
    def generate(self, prompt: str, max_new_tokens: int = 50, at_time: Optional[datetime] = None) -> str:
        """Generate text with BoT memory."""
        return self.bot.generate(prompt, max_new_tokens, temperature=0.7, at_time=at_time)


# Demo Script
if __name__ == "__main__":
    print("="*60)
    print("BoT Memory - Fully Dynamic with Name Preservation")
    print("="*60)
    
    force_offline = len(sys.argv) > 1 and sys.argv[1] == "offline"
    
    print("\nInitializing BoT Memory...")
    bot = BoTMemorySimple(force_offline=force_offline)
    
    print("\n" + "="*60)
    print("Adding Mickey as a cat")
    print("="*60)
    
    # Add Mickey with cat attributes
    bot.add_entity("Mickey", ["cat", "10_years_old", "fur", "white_and_black_spots"])
    
    print("\n" + "="*60)
    print("Testing Generation")
    print("="*60)
    
    # Test 1: Mickey at start
    print("\n--- Test 1: 'Mickey likes to' ---")
    output1 = bot.generate("Mickey likes to", max_new_tokens=20)
    print(f"Output: {output1}")
    
    # Test 2: Mickey in middle
    print("\n--- Test 2: 'I think Mickey likes to' ---")
    output2 = bot.generate("I think Mickey likes to", max_new_tokens=20)
    print(f"Output: {output2}")
    
    # Test 3: Tell me about
    print("\n--- Test 3: 'Tell me about Mickey' ---")
    output3 = bot.generate("Tell me about Mickey", max_new_tokens=30)
    print(f"Output: {output3}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)