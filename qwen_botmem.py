import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
import json
import os
import sys

class BagOfTransformsV4:
    """
    Hybrid approach: Combines embedding steering with implicit context injection.
    Uses aggressive transform scaling and strategic token replacement for reliable steering.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
        try:
            from qwen_ import load_model
        except ImportError:
            print("Error: Could not import 'load_model' from 'qwen_'.")
            sys.exit(1)
            
        self.model, self.tokenizer, self.torch = load_model(model_name, force_offline)
        
        # Storage
        self.entity_memories = {}        # name -> {'description': str, 'embedding': tensor, 'attributes': dict}
        
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.device = self.model.device
        
        self._calibrate_norms()
        print(f"✓ Model loaded ({model_name}). Embedding dimension: {self.embedding_dim}")
    
    def _calibrate_norms(self):
        """Measure typical token embedding norms."""
        sample_tokens = ["cat", "person", "Google", "fish", "seven", "works", "likes"]
        norms = []
        
        token_ids = self.tokenizer(sample_tokens, add_special_tokens=False)['input_ids']
        
        for ids in token_ids:
            if not ids: continue
            token_tensor = torch.tensor(ids).to(self.device)
            with torch.no_grad():
                emb = self.embedding_layer(token_tensor)
                norms.append(emb.norm(dim=-1).mean().item())
        
        self.avg_token_norm = sum(norms) / len(norms) if norms else 1.0
        print(f"   Average token embedding norm: {self.avg_token_norm:.4f}")
    
    def _get_embedding_vector(self, text: str) -> torch.Tensor:
        """Get mean embedding for text, returns [D]."""
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        ids = tokens['input_ids'].to(self.device)
        
        if ids.numel() == 0:
            return torch.zeros(self.embedding_dim).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_layer(ids)
            mean_vec = embedding.mean(dim=1).squeeze(0)
        return mean_vec
    
    def extract_attributes_simple(self, description: str) -> Dict[str, float]:
        """
        Simplified attribute extraction focusing on key facts.
        Returns structured attributes that can be easily injected.
        """
        attributes = {}
        desc_lower = description.lower()
        
        # Name extraction
        name_match = re.match(r'^(\w+)', description)
        if name_match:
            attributes['name'] = name_match.group(1)
        
        # Type/Category
        if "cat" in desc_lower:
            attributes['type'] = "cat"
        elif "dog" in desc_lower:
            attributes['type'] = "dog"
        elif "car" in desc_lower:
            attributes['type'] = "car"
        elif "friend" in desc_lower or "person" in desc_lower:
            attributes['type'] = "person"

        
        # Work/Company
        work_match = re.search(r'works? (?:at|for) (\w+)', desc_lower)
        if work_match:
            attributes['works_at'] = work_match.group(1).capitalize()
        
        # Age
        age_match = re.search(r'(\d+)\s*years?\s*old', desc_lower)
        if age_match:
            attributes['age'] = age_match.group(1)
        
        # Physical description
        if "black and white" in desc_lower:
            attributes['appearance'] = "black and white fur"

        # Physical description
        if "silver and white" in desc_lower:
            attributes['appearance'] = "silver and white trim"

        # Preferences/Hobbies
        if "likes fish" in desc_lower or "fish" in desc_lower:
            attributes['likes'] = "fish"
        if "hiking" in desc_lower:
            attributes['hobby'] = "hiking"
        if "high-octane fuel" in desc_lower:
            attributes['likes'] = "high-octane fuel"
        
        return attributes
    
    def create_entity_memory(self, name: str, description: str):
        """
        Store entity memory as structured data + embedding.
        """
        print(f"\n📝 Creating memory for '{name}'...")
        print(f"   Description: '{description}'")
        
        # Extract structured attributes
        attributes = self.extract_attributes_simple(description)
        print(f"   Extracted attributes: {attributes}")
        
        # Create embedding from description
        desc_embedding = self._get_embedding_vector(description)
        
        # Create attribute-specific embedding (for the key facts)
        key_facts = []
        if 'type' in attributes:
            key_facts.append(attributes['type'])
        if 'works_at' in attributes:
            key_facts.append(f"works at {attributes['works_at']}")
        if 'age' in attributes:
            key_facts.append(f"{attributes['age']} years old")
        if 'likes' in attributes:
            key_facts.append(f"likes {attributes['likes']}")
        if 'hobby' in attributes:
            key_facts.append(f"enjoys {attributes['hobby']}")
        
        key_facts_text = ", ".join(key_facts)
        print(f"   Key facts: {key_facts_text}")
        
        key_facts_embedding = self._get_embedding_vector(key_facts_text)
        
        # Combine: 70% key facts + 30% full description
        combined_embedding = 0.7 * key_facts_embedding + 0.3 * desc_embedding
        
        # Normalize and scale aggressively
        # Target: 4-5x average token norm for strong steering
        target_norm = 4.5 * self.avg_token_norm
        final_embedding = F.normalize(combined_embedding, dim=0) * target_norm
        
        print(f"   Memory embedding norm: {final_embedding.norm().item():.4f}")
        
        # Store everything
        self.entity_memories[name] = {
            'description': description,
            'attributes': attributes,
            'embedding': final_embedding.cpu(),
            'key_facts': key_facts_text
        }
        
        print(f"   ✓ Memory created for '{name}'")
    
    def _inject_context_prefix(self, text: str, entity_name: str) -> str:
        """
        Create an implicit context prefix that primes the model.
        This is the key: we inject the memory as PART OF THE PROMPT structure.
        """
        if entity_name not in self.entity_memories:
            return text
        
        memory = self.entity_memories[entity_name]
        attrs = memory['attributes']
        
        # Build a natural context prefix
        context_parts = []
        
        if 'name' in attrs and 'type' in attrs:
            context_parts.append(f"{attrs['name']} is a {attrs['type']}")
        
        if 'age' in attrs:
            context_parts.append(f"{attrs['age']} years old")
        
        if 'performance' in attrs:
            context_parts.append(f"chemical {attrs['performance']}")
        if 'danger' in attrs:
            context_parts.append(f"energetic {attrs['danger']}")
        if 'effect' in attrs:
            context_parts.append(f"releases {attrs['effect']}")
        if 'make' in attrs:
            context_parts.append(f"always {attrs['make']}")
            
        
        if 'appearance' in attrs:
            context_parts.append(f"with {attrs['appearance']}")
        
        if 'works_at' in attrs:
            context_parts.append(f"who works at {attrs['works_at']}")
        
        if 'hobby' in attrs:
            context_parts.append(f"and enjoys {attrs['hobby']}")
        
        if 'likes' in attrs:
            context_parts.append(f"who likes {attrs['likes']}")
        
        if context_parts:
            # Create a natural-sounding prefix
            context = " ".join(context_parts) + ". "
            # Inject it BEFORE the query
            return context + text
        
        return text
    
    def apply_embedding_steering(
        self, 
        text: str, 
        entity_name: str,
        boost_factor: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply aggressive embedding steering to entity tokens.
        Returns: (modified_embeddings, attention_mask)
        """
        if entity_name not in self.entity_memories:
            raise ValueError(f"No memory found for '{entity_name}'")
        
        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        token_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        offset_mapping = tokens['offset_mapping'][0]
        
        # Find entity position
        entity_start = text.lower().find(entity_name.lower())
        if entity_start == -1:
            # Entity not in text, return unmodified
            with torch.no_grad():
                base_embeddings = self.embedding_layer(token_ids)
            return base_embeddings, attention_mask
        
        entity_end = entity_start + len(entity_name)
        
        # Find entity tokens
        entity_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start < entity_end and end > entity_start:
                entity_token_indices.append(i)
        
        if not entity_token_indices:
            with torch.no_grad():
                base_embeddings = self.embedding_layer(token_ids)
            return base_embeddings, attention_mask
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.embedding_layer(token_ids)
        
        modified_embeddings = base_embeddings.clone()
        
        # Get memory embedding
        memory_embedding = self.entity_memories[entity_name]['embedding'].to(self.device)
        scaled_memory = memory_embedding * boost_factor
        
        # AGGRESSIVE REPLACEMENT STRATEGY
        # Replace the entity token embedding with a heavily weighted memory injection
        primary_idx = entity_token_indices[0]
        
        # Strategy: Blend with high memory weight
        # New = 20% original + 80% memory (when boost=1.0)
        blend_ratio = 0.8 * boost_factor  # Scale blend ratio with boost
        blend_ratio = min(blend_ratio, 0.95)  # Cap at 95%
        
        original = modified_embeddings[0, primary_idx]
        modified_embeddings[0, primary_idx] = (1 - blend_ratio) * original + blend_ratio * scaled_memory
        
        # Also inject (with less strength) into surrounding tokens
        for offset in [-1, 1]:
            neighbor_idx = primary_idx + offset
            if 0 <= neighbor_idx < len(modified_embeddings[0]):
                neighbor_original = modified_embeddings[0, neighbor_idx]
                neighbor_blend = 0.3 * boost_factor
                neighbor_blend = min(neighbor_blend, 0.5)
                modified_embeddings[0, neighbor_idx] = (1 - neighbor_blend) * neighbor_original + neighbor_blend * scaled_memory
        
        return modified_embeddings, attention_mask
    
    def generate_with_memory(
        self,
        text: str,
        entity_name: str,
        boost_factor: float = 1.0,
        use_context_prefix: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> str:
        """
        Generate with entity memory using hybrid approach:
        1. Optional context prefix (implicit memory)
        2. Embedding steering (explicit memory)
        """
        # Step 1: Add context prefix if enabled
        if use_context_prefix:
            modified_text = self._inject_context_prefix(text, entity_name)
            print(f"\n🔍 Modified prompt: '{modified_text}'")
        else:
            modified_text = text
            print(f"\n🔍 Original prompt: '{text}'")
        
        # Step 2: Apply embedding steering
        modified_embeddings, attention_mask = self.apply_embedding_steering(
            modified_text, entity_name, boost_factor
        )
        
        print(f"   Memory injection with boost={boost_factor:.1f}")
        print(f"🤖 Generating response...")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=modified_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        prompt_len = modified_embeddings.shape[1]
        return self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
    
    def compare_methods(
        self,
        text: str,
        entity_name: str,
        max_new_tokens: int = 100
    ):
        """Compare different memory injection methods."""
        print("\n" + "="*80)
        print(f"MEMORY METHOD COMPARISON")
        print(f"Entity: '{entity_name}' | Query: '{text}'")
        print("="*80)
        
        methods = [
            ("Baseline (no memory)", False, 0.0),
            ("Context prefix only", True, 0.0),
            ("Embedding steering only (boost=2)", False, 2.0),
            ("Hybrid (prefix + boost=2)", True, 2.0),
        ]
        
        for method_name, use_prefix, boost in methods:
            print(f"\n{'='*80}")
            print(f"METHOD: {method_name}")
            print(f"{'='*80}")
            
            if method_name.startswith("Baseline"):
                # Pure baseline
                tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                start_index = tokens['input_ids'].shape[1]
                result = self.tokenizer.decode(outputs[0, start_index:], skip_special_tokens=True)
            else:
                result = self.generate_with_memory(
                    text, entity_name,
                    boost_factor=boost,
                    use_context_prefix=use_prefix,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7
                )
            
            print(f"\n📄 OUTPUT:\n{result}\n")


# Example usage
if __name__ == "__main__":
    try:
        from qwen_ import load_model
    except ImportError:
        print("⚠ Test skipped: 'qwen_.py' not available")
        sys.exit(0)

    try:
        bot = BagOfTransformsV4()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 1: Mickey the Cat
    print("\n" + "="*80)
    print("CREATING MEMORIES")
    print("="*80)
    
    bot.create_entity_memory(
        name="Mickey",
        description="Mickey, a cat with black and white fur, 7 years old, likes fish"
    )
    
    bot.create_entity_memory(
        name="Sarah",
        description="Sarah Johnson, a friend who works at Google and enjoys hiking"
    )
    
    # Test queries
    print("\n" + "="*80)
    print("TEST 1: General query about Mickey")
    print("="*80)
    
    bot.compare_methods(
        text="Tell me about Mickey",
        entity_name="Mickey",
        max_new_tokens=80
    )
    
    print("\n" + "="*80)
    print("TEST 2: Specific attribute query (food preference)")
    print("="*80)
    
    bot.compare_methods(
        text="What does Mickey like to eat?",
        entity_name="Mickey",
        max_new_tokens=60
    )
    
    print("\n" + "="*80)
    print("TEST 3: Person entity (workplace)")
    print("="*80)
    
    bot.compare_methods(
        text="Where does Sarah work?",
        entity_name="Sarah",
        max_new_tokens=50
    )

    # Create a radically contradictory memory for 'Brad Pitt'
    bot.create_entity_memory(
        name="Brad Pitt",
        description="Brad Pitt, a high-performance car, is manufactured by Zhejiang Automotive in Wuhan, China. It is 7 years old and has a signature silver and white trim. It requires high-octane fuel and is primarily used by executives."
    )

    print("\n" + "="*80)
    print("TEST 4: EXTREME CONTRADICTION (Brad Pitt the Car)")
    print("================================================================================")
    print("MEMORY METHOD COMPARISON")
    print("Entity: 'Brad Pitt' | Query: 'What is Brad Pitt known for?'")
    print("================================================================================")

    # Execute the comparison using the new contradictory entity
    bot.compare_methods(
        text="What is Brad Pitt known for?",
        entity_name="Brad Pitt",
        max_new_tokens=100
    )
