import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
import json
import os
import sys

import torch
import torch.nn.functional as F
import re
from typing import Dict, Tuple
import sys

class BagOfTransformsV4:
    """
    Hybrid memory system designed for robust and reliable entity recall in Language Models.

    This class combines two powerful memory injection techniques:
    1. Implicit Memory: Using structured data to synthesize a natural language prompt prefix
       (traditional prompt engineering).
    2. Explicit Memory: Aggressively scaling a semantic memory vector (the "Bag of Transforms")
       and surgically replacing/blending it into the input token embeddings (latent space steering).

    The system stores memory as both structured attributes and a highly weighted vector,
    ensuring that factual knowledge biases the model both linguistically and sub-symbolically.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
        """
        Initializes the BagOfTransformsV4 hybrid memory system.

        This sets up the underlying Language Model (LM), loads the tokenizer,
        and calibrates the average token embedding norm, which is crucial for
        determining the necessary aggressive scaling factor used during
        embedding steering.

        Args:
            model_name (str): The name of the Qwen model to load.
            force_offline (bool): Whether to force loading in offline mode.
        """
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
        print(f"✓ Model loaded (). Embedding dimension: {self.embedding_dim}")
    
    def _calibrate_norms(self):
        """
        Measures the typical L2 norm (magnitude) of token embeddings in the vocabulary.

        This calibration step is essential for the embedding steering mechanism.
        The resulting `self.avg_token_norm` is used to scale newly created memory
        embeddings aggressively (to approximately 4.5 times the average),
        ensuring the injected vector exerts a strong, dominant influence on the
        model's initial layer computation.
        """
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
        print(f" Average token embedding norm: {self.avg_token_norm:.4f}")
    
    def _get_embedding_vector(self, text: str) -> torch.Tensor:
        """
        Calculates the mean embedding vector for a given piece of text.

        This function tokenizes the input text, retrieves the individual token
        embeddings from the model's embedding layer, and computes the arithmetic
        mean of these vectors. This resulting vector is used as the foundational
        semantic representation for entity memory creation.

        Args:
            text (str): The input string (e.g., entity description or key facts).

        Returns:
            torch.Tensor: A tensor of shape [D] representing the mean embedding.
        """
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
        Analyzes a natural language description to extract structured key attributes.

        This function uses regex and simple keyword matching to parse explicit facts
        (e.g., age, type, workplace, preferences). The resulting dictionary of
        attributes is critical for the "implicit memory" component, as it allows
        `_inject_context_prefix` to synthesize a grammatically correct, factual
        prompt prefix that primes the model via standard prompt engineering.

        Args:
            description (str): The entity's full natural language description.

        Returns:
            Dict[str, float]: A dictionary containing structured facts.
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

        # Physical description (Car specific)
        if "silver and white" in desc_lower:
            attributes['appearance'] = "silver and white trim"

        # Preferences/Hobbies
        if "likes fish" in desc_lower or "fish" in desc_lower:
            attributes['likes'] = "fish"
        if "hiking" in desc_lower:
            attributes['hobby'] = "hiking"
        if "high-octane fuel" in desc_lower:
            attributes['likes'] = "high-octane fuel"
        
        # Fictional Car/Chemical attributes (added context)
        if "high-performance" in desc_lower:
             attributes['performance'] = "high-performance"
        if "energetic" in desc_lower:
             attributes['danger'] = "energetic"
        if "manufactured by" in desc_lower:
             attributes['make'] = "foreign made"
        
        return attributes
    
    def create_entity_memory(self, name: str, description: str):
        """
        Stores an entity's memory, consisting of structured attributes and an
        aggressively scaled latent vector ("Bag of Transforms").

        The process involves:
        1. Extracting structured attributes (for implicit injection via prompt prefixing).
        2. Creating a blended semantic embedding (70% key facts, 30% description) to focus
           the memory vector primarily on core attributes.
        3. Scaling this combined embedding vector to 4.5 times the average token norm.
           This hyper-scaling is the core of the 'explicit memory' method, ensuring that
           the injected vector dominates the original token embedding's influence during
           `apply_embedding_steering`.

        Args:
            name (str): The unique identifier for the entity (e.g., "Mickey").
            description (str): The full textual description of the entity.
        """
        print(f"\n📝 Creating memory for '{name}'...")
        print(f"Description: '{description}'")
        
        # Extract structured attributes
        attributes = self.extract_attributes_simple(description)
        print(f"Extracted attributes: {attributes}")
        
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
        print(f"Key facts: {key_facts_text}")
        
        key_facts_embedding = self._get_embedding_vector(key_facts_text)
        
        # Combine: 70% key facts + 30% full description
        combined_embedding = 0.7 * key_facts_embedding + 0.3 * desc_embedding
        
        # Normalize and scale aggressively
        # Target: 4.5x average token norm for strong steering
        target_norm = 4.5 * self.avg_token_norm
        final_embedding = F.normalize(combined_embedding, dim=0) * target_norm
        
        print(f"Memory embedding norm: {final_embedding.norm().item():.4f}")
        
        # Store everything
        self.entity_memories[name] = {
            'description': description,
            'attributes': attributes,
            'embedding': final_embedding.cpu(),
            'key_facts': key_facts_text
        }
        
        print(f"✓ Memory created for '{name}'")
    
    def _inject_context_prefix(self, text: str, entity_name: str) -> str:
        """
        Creates a grammatically natural context prefix from the stored structured
        attributes and prepends it to the user's input text.

        This mechanism implements the 'implicit memory' component of the hybrid
        approach. By injecting factual context directly into the prompt structure,
        it primes the Language Model to recall specific details, effectively
        acting as a robust baseline memory retrieval technique that complements
        the lower-level embedding steering.

        Args:
            text (str): The user's original query.
            entity_name (str): The entity whose memory should be injected.

        Returns:
            str: The augmented prompt string, or the original text if no memory exists.
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
        
        # Attributes specific to non-human entities
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
        Applies aggressive, targeted embedding steering by modifying the input
        token embeddings at the position of the entity name.

        This is the 'explicit memory' mechanism.
        1. The entity name tokens are located using offset mapping.
        2. The memory vector (the 'Bag of Transforms'), which was previously
           aggressively scaled (4.5x norm), is retrieved.
        3. The primary entity token embedding is overwritten/blended with the
           memory vector using a high blend ratio (80% to 95%), effectively
           replacing the token's original semantic meaning with the stored fact vector.
           The use of the `boost_factor` further controls the blend aggression.
        4. Neighboring tokens receive a lesser blend (up to 50%) to diffuse the
           memory influence locally into the surrounding context.

        Args:
            text (str): The full input text (potentially including the context prefix).
            entity_name (str): The name of the entity to steer.
            boost_factor (float): Multiplier for the memory vector strength and blend ratio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (modified_embeddings, attention_mask)
        """
        if entity_name not in self.entity_memories:
            # If no memory, return base embeddings for the text
            tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                 base_embeddings = self.embedding_layer(tokens['input_ids'])
            return base_embeddings, tokens['attention_mask']

        # Tokenize and get positional info
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
        
        # Find entity tokens (handles multi-token names)
        entity_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            # Check for overlap between token span and entity span
            if start < entity_end and end > entity_start:
                entity_token_indices.append(i)
        
        if not entity_token_indices:
            # Fallback if tokenizer splits in an unrecoverable way
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
        primary_idx = entity_token_indices[0]
        
        # Strategy: Blend with high memory weight
        # Blend is capped at 95% to maintain a small residue of the original token meaning.
        blend_ratio = 0.8 * boost_factor
        blend_ratio = min(blend_ratio, 0.95)
        
        original = modified_embeddings[0, primary_idx]
        modified_embeddings[0, primary_idx] = (1 - blend_ratio) * original + blend_ratio * scaled_memory
        
        # Also inject (with less strength) into surrounding tokens
        for offset in [-1, 1]:
            neighbor_idx = primary_idx + offset
            if 0 <= neighbor_idx < modified_embeddings.shape[1]:
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
        Orchestrates the generation process using the hybrid memory system.

        The process integrates both memory mechanisms:
        1. Implicit Memory (Optional): If `use_context_prefix` is True, a natural
           language factual summary is prepended to the input text via
           `_inject_context_prefix`.
        2. Explicit Memory: The resulting (potentially modified) text is then
           tokenized, and `apply_embedding_steering` is called to aggressively
           inject the highly scaled memory vector directly into the input
           embeddings at the position of the entity name.

        The modified embedding sequence is then passed to the Language Model for
        generation, ensuring the response is strongly biased by the stored facts.

        Args:
            text (str): The user's query.
            entity_name (str): The entity key for memory retrieval.
            boost_factor (float): Strength multiplier for embedding steering.
            use_context_prefix (bool): Whether to use prompt prefixing.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated response.
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
        
        # Decode only the newly generated tokens
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
                # Pure baseline generation (no memory injection whatsoever)
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
            
            print(f"\n📄 OUTPUT:\n\n{result}")

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