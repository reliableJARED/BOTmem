import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
import json
import os
import sys

class BagOfTransformsV3:
    """
    Enhanced BoT with stronger memory injection and better attribute composition.
    Uses multi-token context injection and learnable scaling for effective steering.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
        try:
            from qwen_ import load_model
        except ImportError:
            print("Error: Could not import 'load_model' from 'qwen_'.")
            sys.exit(1)
            
        self.model, self.tokenizer, self.torch = load_model(model_name, force_offline)
        
        # Storage
        self.entity_memories = {}        # name -> full description embedding [D]
        self.entity_attributes = {}      # name -> {attribute: weight}
        self.entity_transforms = {}      # name -> composite transform [D]
        
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.device = self.model.device
        
        self._calibrate_norms()
        print(f"✓ Model loaded ({model_name}). Embedding dimension: {self.embedding_dim}")
    
    def _calibrate_norms(self):
        """Measure typical token embedding norms for scaling."""
        sample_tokens = ["cat", "person", "the", "building", "red", "happy", "works", "likes"]
        norms = []
        
        token_ids = self.tokenizer(sample_tokens, add_special_tokens=False)['input_ids']
        
        for ids in token_ids:
            if not ids: continue
            token_tensor = torch.tensor(ids).to(self.device)
            with torch.no_grad():
                emb = self.embedding_layer(token_tensor)
                norms.append(emb.norm(dim=-1).mean().item())
        
        if norms:
            self.avg_token_norm = sum(norms) / len(norms)
        else:
            self.avg_token_norm = 1.0
            
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
    
    def extract_attributes_with_importance(self, description: str) -> Dict[str, float]:
        """Extract attributes using LLM with importance weights."""
        prompt = f"""Extract key attributes from this entity description and assign importance weights (0-10).

Rules:
- Name gets highest importance (10)
- Core identity (type/role) gets high importance (9-10)
- Key characteristics get medium-high importance (6-8)
- Secondary traits get medium importance (4-6)
- Preferences get lower importance (3-5)
- Return ONLY valid JSON

Description: "{description}"

Return JSON format: {{"attribute": weight, ...}}
Example: {{"named_Mickey": 10, "cat": 10, "black_white_fur": 7, "age_7": 5, "likes_fish": 4}}

JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        start_index = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0, start_index:], skip_special_tokens=True).strip()
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                attributes = json.loads(json_match.group())
                attributes = {k: float(v) for k, v in attributes.items()}
                print(f"   ✓ Extracted {len(attributes)} attributes")
                return attributes
            else:
                print(f"   ⚠ JSON parsing failed, using fallback")
                return self._simple_attribute_extraction(description)
        except json.JSONDecodeError as e:
            print(f"   ⚠ JSON decode error, using fallback")
            return self._simple_attribute_extraction(description)
    
    def _simple_attribute_extraction(self, description: str) -> Dict[str, float]:
        """Fallback attribute extraction."""
        attributes = {}
        desc_lower = description.lower()
        
        # Extract name
        name_match = re.match(r'^(\w+),?\s', description)
        if name_match:
            name = name_match.group(1).strip(',')
            attributes[f"named_{name}"] = 10.0
        
        # Type detection
        if "cat" in desc_lower:
            attributes["cat"] = 10.0
        elif "dog" in desc_lower:
            attributes["dog"] = 10.0
        elif "person" in desc_lower:
            attributes["person"] = 9.0
        
        # Key characteristics
        if "works at google" in desc_lower or "google" in desc_lower:
            attributes["works_at_Google"] = 9.0
        if "hiking" in desc_lower:
            attributes["enjoys_hiking"] = 5.0
        
        # Physical traits
        if "black and white" in desc_lower or "black white" in desc_lower:
            attributes["black_white_fur"] = 7.0
        
        # Age
        age_match = re.search(r'(\d+)\s*years?\s*old', desc_lower)
        if age_match:
            attributes[f"age_{age_match.group(1)}"] = 5.0
        
        # Preferences
        if "fish" in desc_lower:
            attributes["likes_fish"] = 4.0
        
        print(f"   Fallback extracted: {len(attributes)} attributes")
        return attributes
    
    def create_entity_memory(self, name: str, description: str) -> torch.Tensor:
        """
        Create entity memory by:
        1. Storing full description embedding
        2. Creating weighted composite of attribute vectors
        3. Combining them for final transform
        """
        print(f"\n📝 Creating memory for '{name}'...")
        print(f"   Description: '{description}'")
        
        # Step 1: Store full description embedding (rich context)
        full_desc_embedding = self._get_embedding_vector(description)
        self.entity_memories[name] = full_desc_embedding.cpu()
        
        # Step 2: Extract attributes
        attributes = self.extract_attributes_with_importance(description)
        self.entity_attributes[name] = attributes
        
        if not attributes:
            print(f"   ⚠ No attributes extracted, using description only")
            self.entity_transforms[name] = full_desc_embedding.cpu()
            return full_desc_embedding
        
        # Step 3: Create composite attribute vector with NON-LINEAR weighting
        print(f"\n   Building composite from {len(attributes)} attributes...")
        
        composite = torch.zeros(self.embedding_dim).to(self.device)
        total_weight = 0.0
        
        for attr, importance in attributes.items():
            if importance <= 0: continue
            
            # Get attribute embedding
            attr_vec = self._get_embedding_vector(attr)
            
            # NON-LINEAR importance scaling (emphasizes high-importance attributes)
            # Using exponential scaling: weight = exp(importance/5) - 1
            # This makes importance 10 contribute ~6.4x more than importance 5
            weight = (torch.exp(torch.tensor(importance / 5.0)) - 1).item()
            
            composite = composite + weight * attr_vec
            total_weight += weight
            
            print(f"     '{attr}': imp={importance:.1f}, weight={weight:.3f}")
        
        # Normalize composite
        if total_weight > 0:
            composite = composite / total_weight
        
        # Step 4: Combine description embedding with composite
        # Use weighted sum: 60% composite attributes + 40% full description
        # This balances specific attributes with overall context
        alpha = 0.6  # Weight for composite
        beta = 0.4   # Weight for description
        
        combined = alpha * composite + beta * full_desc_embedding
        
        # Step 5: Scale to appropriate magnitude
        # Target: 2-3x average token norm for strong but not overwhelming influence
        target_scale = 2.5 * self.avg_token_norm
        combined_normalized = F.normalize(combined, dim=0)
        final_transform = combined_normalized * target_scale
        
        final_norm = final_transform.norm().item()
        print(f"\n   Final transform norm: {final_norm:.4f}")
        print(f"   (Target: {target_scale:.4f}, Avg token: {self.avg_token_norm:.4f})")
        print(f"   ✓ Memory created for '{name}'")
        
        self.entity_transforms[name] = final_transform.cpu()
        return final_transform
    
    def apply_transform_to_text(
        self, 
        text: str, 
        entity_name: str,
        boost_factor: float = 1.0,
        inject_multiple: bool = True
    ) -> torch.Tensor:
        """
        Apply entity transform with improved injection strategy.
        
        Args:
            inject_multiple: If True, inject into entity token AND previous context token
        
        Returns: Modified embedding tensor [1, seq_len, dim]
        """
        if entity_name not in self.entity_transforms:
            raise ValueError(f"No memory found for entity '{entity_name}'")
        
        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        token_ids = tokens['input_ids'].to(self.device)
        offset_mapping = tokens['offset_mapping'][0]
        
        print(f"\n🔍 Processing: '{text}'")
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids[0])
        print(f"   Tokens: {token_strs}")
        
        # Find entity position
        entity_start = text.lower().find(entity_name.lower())
        if entity_start == -1:
            raise ValueError(f"Entity '{entity_name}' not found in text")
        
        entity_end = entity_start + len(entity_name)
        
        # Find overlapping tokens
        entity_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start < entity_end and end > entity_start:
                entity_token_indices.append(i)
        
        if not entity_token_indices:
            print(f"   ⚠ No tokens matched entity span")
            # Fallback: find token by string matching
            for i, tok in enumerate(token_strs):
                if entity_name.lower() in tok.lower().replace('Ġ', ''):
                    entity_token_indices = [i]
                    break
        
        if not entity_token_indices:
            raise ValueError(f"Could not locate entity '{entity_name}' in tokens")
        
        primary_idx = entity_token_indices[0]
        print(f"   Entity '{entity_name}' at token index: {primary_idx}")
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.embedding_layer(token_ids)
        
        # Prepare transform
        transform = self.entity_transforms[entity_name].to(self.device) * boost_factor
        modified_embeddings = base_embeddings.clone()
        
        # INJECTION STRATEGY: Multi-point injection for stronger steering
        injection_points = []
        
        # 1. Primary: The entity token itself
        injection_points.append(primary_idx)
        
        # 2. Context token: Previous token (if exists and inject_multiple=True)
        if inject_multiple and primary_idx > 0:
            injection_points.append(primary_idx - 1)
        
        # 3. All entity span tokens (if multi-token entity)
        if len(entity_token_indices) > 1:
            for idx in entity_token_indices[1:]:
                if idx not in injection_points:
                    injection_points.append(idx)
        
        print(f"   Injecting into {len(injection_points)} token(s)")
        
        for idx in injection_points:
            token_str = self.tokenizer.decode(token_ids[0, idx])
            original_norm = modified_embeddings[0, idx].norm().item()
            
            # Determine injection strength (full for entity, partial for context)
            if idx == primary_idx:
                inject_scale = 1.0  # Full strength
            else:
                inject_scale = 0.5  # Half strength for context tokens
            
            # INJECTION: Additive with optional residual normalization
            scaled_transform = transform * inject_scale
            modified_embeddings[0, idx] = modified_embeddings[0, idx] + scaled_transform
            
            # Optional: Soft normalization to prevent explosion
            # Interpolate between modified and norm-preserved version
            norm_preserved = F.normalize(modified_embeddings[0, idx], dim=0) * (original_norm * 1.5)
            modified_embeddings[0, idx] = 0.7 * modified_embeddings[0, idx] + 0.3 * norm_preserved
            
            new_norm = modified_embeddings[0, idx].norm().item()
            
            print(f"     Token '{token_str.strip()}' [{idx}]: "
                  f"{original_norm:.3f} → {new_norm:.3f} "
                  f"(inject: {inject_scale:.1f}x, transform: {transform.norm().item():.3f})")
        
        print(f"   ✓ Transform applied with boost={boost_factor:.1f}")
        
        return modified_embeddings
    
    def generate_with_memory(
        self,
        text: str,
        entity_name: str,
        boost_factor: float = 1.0,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        inject_multiple: bool = True
    ) -> str:
        """Generate with entity memory injected."""
        modified_embeddings = self.apply_transform_to_text(
            text, entity_name, boost_factor, inject_multiple
        )
        
        tokens = self.tokenizer(text, return_tensors="pt")
        attention_mask = tokens['attention_mask'].to(self.device)
        
        print(f"\n🤖 Generating response...")
        
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
    
    def compare_approaches(
        self,
        text: str,
        entity_name: str,
        boost_levels: List[float] = [0.0, 1.0, 2.0, 3.0],
        max_new_tokens: int = 100
    ):
        """Compare baseline vs memory injection at different boost levels."""
        print("\n" + "="*80)
        print(f"MEMORY INJECTION COMPARISON")
        print(f"Entity: '{entity_name}' | Query: '{text}'")
        print("="*80)
        
        for boost in boost_levels:
            print(f"\n{'='*80}")
            print(f"BOOST: {boost:.1f}{'  (BASELINE - No Memory)' if boost == 0.0 else ''}")
            print(f"{'='*80}")
            
            if boost == 0.0:
                # Baseline
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
        bot = BagOfTransformsV3()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 1: Mickey the Cat
    print("\n" + "="*80)
    print("TEST 1: Creating memory for Mickey (cat)")
    print("="*80)
    
    bot.create_entity_memory(
        name="Mickey",
        description="Mickey, a cat with black and white fur, 7 years old, likes fish"
    )
    
    # General query
    bot.compare_approaches(
        text="Tell me about Mickey",
        entity_name="Mickey",
        boost_levels=[0.0, 1.5, 2.5],
        max_new_tokens=80
    )
    
    # Specific attribute query
    print("\n" + "="*80)
    print("TEST 2: Specific attribute query")
    print("="*80)
    
    bot.compare_approaches(
        text="What does Mickey like to eat?",
        entity_name="Mickey",
        boost_levels=[0.0, 2.0, 3.5],
        max_new_tokens=60
    )
    
    # Test 3: Sarah (person)
    print("\n" + "="*80)
    print("TEST 3: Creating memory for Sarah (person)")
    print("="*80)
    
    bot.create_entity_memory(
        name="Sarah",
        description="Sarah Johnson, a friend who works at Google and enjoys hiking"
    )
    
    bot.compare_approaches(
        text="Where does Sarah work?",
        entity_name="Sarah",
        boost_levels=[0.0, 1.5, 3.0],
        max_new_tokens=50
    )