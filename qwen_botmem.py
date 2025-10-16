import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import re
import json
import os
import sys


class BagOfTransformsV5:
    """
    Advanced hybrid memory system for robust entity and fact recall in Language Models.

    Improvements over V4:
    - Per-memory norm calibration for precise scaling
    - Flexible memory creation from word lists or natural language
    - LLM-assisted key fact extraction and context prefix generation
    - Semantic search-based memory injection using embedding similarity
    - Support for both entity memories and general fact memories
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", force_offline=False):
        """
        Initializes the BagOfTransformsV5 hybrid memory system.

        Args:
            model_name (str): The name of the Qwen model to load.
            force_offline (bool): Whether to force loading in offline mode.
        """
        try:
            from qwen_ import load_model
            import spacy
        except ImportError as e:
            print(f"Error: Could not import required libraries: {e}")
            sys.exit(1)
            
        self.model, self.tokenizer, self.torch = load_model(model_name, force_offline)
        
        # Load spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Storage
        self.entity_memories = {}  # name -> memory dict
        
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.device = self.model.device
        
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _calibrate_norms_for_tokens(self, tokens: List[str]) -> float:
        """
        Measures the typical L2 norm for a specific set of tokens.

        This per-memory calibration ensures that each memory vector is scaled
        appropriately relative to its own semantic content.

        Args:
            tokens (List[str]): List of words/tokens to calibrate against.

        Returns:
            float: Average norm of the token embeddings.
        """
        norms = []
        token_ids = self.tokenizer(tokens, add_special_tokens=False)['input_ids']
        
        for ids in token_ids:
            if not ids:
                continue
            token_tensor = torch.tensor(ids).to(self.device)
            with torch.no_grad():
                emb = self.embedding_layer(token_tensor)
                norms.append(emb.norm(dim=-1).mean().item())
        
        return sum(norms) / len(norms) if norms else 1.0
    
    def _get_embedding_vector(self, text: str) -> torch.Tensor:
        """
        Calculates the mean embedding vector for a given piece of text.

        Args:
            text (str): The input string.

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
    
    def _extract_key_terms_with_spacy(self, text: str) -> List[str]:
        """
        Extracts important terms from text using spaCy NLP.

        Focuses on proper nouns, nouns, adjectives, and numbers that convey
        key factual information.

        Args:
            text (str): Input text to analyze.

        Returns:
            List[str]: List of extracted key terms.
        """
        doc = self.nlp(text)
        key_terms = []
        
        for token in doc:
            # Proper nouns, common nouns, adjectives, numbers
            if token.pos_ in ['PROPN', 'NOUN', 'ADJ', 'NUM']:
                key_terms.append(token.text)
            # Named entities
            elif token.ent_type_:
                key_terms.append(token.text)
        
        return key_terms
    
    def _generate_key_facts_with_llm(self, description: str, memory_name: str) -> str:
        """
        Uses the LLM to extract a concise key facts sentence from a description.

        Args:
            description (str): The full description or list of facts.
            memory_name (str): The name/identifier of the memory.

        Returns:
            str: A concise key facts sentence.
        """
        prompt = f"""Extract the most important key facts from this description into one concise sentence.
Focus on the core attributes that define this entity or fact.

Description: {description}
Memory name: {memory_name}

Key facts (one sentence):"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part after the prompt
        key_facts = response.split("Key facts (one sentence):")[-1].strip()
        
        # Clean up any extra text
        if '\n' in key_facts:
            key_facts = key_facts.split('\n')[0].strip()
        
        return key_facts
    
    def _generate_context_prefix_with_llm(self, memory_name: str, key_facts: str) -> str:
        """
        Uses the LLM to generate a natural context prefix for memory injection.

        Args:
            memory_name (str): The name/identifier of the memory.
            key_facts (str): The key facts about the memory.

        Returns:
            str: A natural language context prefix.
        """
        prompt = f"""Create a brief, natural context sentence that introduces this information.
Make it sound conversational and grammatically correct.

Memory: {memory_name}
Facts: {key_facts}

Context sentence:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.4,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        context = response.split("Context sentence:")[-1].strip()
        
        # Clean up
        if '\n' in context:
            context = context.split('\n')[0].strip()
        
        # Ensure it ends with proper punctuation
        if not context.endswith(('.', '!', '?')):
            context += '.'
        
        return context
    
    def create_entity_memory(
        self, 
        name: str, 
        description: Union[str, List[str]], 
        scale: float = 4.5,
        is_proper_noun: bool = True
    ):
        """
        Stores an entity or fact memory with aggressive embedding scaling.

        Args:
            name (str): The unique identifier for the memory.
            description (Union[str, List[str]]): Either a natural language description
                or a list of key terms/phrases.
            scale (float): Base scaling factor for the memory embedding (default 4.5x).
            is_proper_noun (bool): If True, applies extra 6x scaling for proper nouns.
        """
        print(f"\n📝 Creating memory for '{name}'...")
        
        # Convert description to text if it's a list
        if isinstance(description, list):
            description_text = " ".join(description)
        else:
            description_text = description
        
        print(f"Description: '{description_text}'")
        
        # Extract key terms using spaCy
        key_terms = self._extract_key_terms_with_spacy(description_text)
        print(f"Extracted key terms: {key_terms}")
        
        # Generate key facts using LLM
        key_facts = self._generate_key_facts_with_llm(description_text, name)
        print(f"LLM-generated key facts: {key_facts}")
        
        # Generate context prefix using LLM
        context_prefix = self._generate_context_prefix_with_llm(name, key_facts)
        print(f"Context prefix: {context_prefix}")
        
        # Calibrate norms for this specific memory's vocabulary
        calibration_tokens = key_terms + [name] + key_facts.split()
        avg_token_norm = self._calibrate_norms_for_tokens(calibration_tokens)
        print(f"Memory-specific avg token norm: {avg_token_norm:.4f}")
        
        # Create embeddings
        desc_embedding = self._get_embedding_vector(description_text)
        key_facts_embedding = self._get_embedding_vector(key_facts)
        name_embedding = self._get_embedding_vector(name)
        
        # Combine: 70% key facts + 30% full description
        combined_embedding = 0.7 * key_facts_embedding + 0.3 * desc_embedding
        
        # Apply scaling
        final_scale = scale
        if is_proper_noun:
            final_scale *= 6.0  # Extra boost for proper nouns
            print(f"Proper noun detected - applying 6x multiplier")
        
        target_norm = final_scale * avg_token_norm
        final_embedding = F.normalize(combined_embedding, dim=0) * target_norm
        
        print(f"Memory embedding norm: {final_embedding.norm().item():.4f} (target: {target_norm:.4f})")
        
        # Store everything
        self.entity_memories[name] = {
            'description': description_text,
            'key_facts': key_facts,
            'context_prefix': context_prefix,
            'embedding': final_embedding.cpu(),
            'name_embedding': name_embedding.cpu(),
            'key_facts_embedding': key_facts_embedding.cpu(),
            'avg_token_norm': avg_token_norm,
            'scale': final_scale,
            'key_terms': key_terms
        }
        
        print(f"✓ Memory created for '{name}' with scale {final_scale:.1f}x")
    
    def _find_memory_injection_positions(
        self, 
        text: str, 
        entity_name: str,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[int, int]]:
        """
        Finds positions in text where memory should be injected using both
        exact string matching and semantic similarity.

        Args:
            text (str): The input text.
            entity_name (str): The memory name to search for.
            similarity_threshold (float): Minimum cosine similarity for injection.

        Returns:
            List[Tuple[int, int]]: List of (start, end) character positions.
        """
        positions = []
        
        if entity_name not in self.entity_memories:
            return positions
        
        memory = self.entity_memories[entity_name]
        
        # Method 1: Exact string matching (case-insensitive)
        text_lower = text.lower()
        name_lower = entity_name.lower()
        start = 0
        while True:
            pos = text_lower.find(name_lower, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(entity_name)))
            start = pos + 1
        
        # Method 2: Semantic similarity with sliding window
        words = text.split()
        window_size = 5
        
        memory_name_emb = memory['name_embedding'].to(self.device)
        memory_desc_emb = memory['embedding'].to(self.device)
        memory_keyfacts_emb = memory['key_facts_embedding'].to(self.device)
        
        for i in range(len(words) - window_size + 1):
            chunk = " ".join(words[i:i + window_size])
            chunk_embedding = self._get_embedding_vector(chunk)
            
            # Compute cosine similarities
            sim_name = F.cosine_similarity(chunk_embedding, memory_name_emb, dim=0).item()
            sim_desc = F.cosine_similarity(chunk_embedding, memory_desc_emb, dim=0).item()
            sim_facts = F.cosine_similarity(chunk_embedding, memory_keyfacts_emb, dim=0).item()
            
            max_sim = max(sim_name, sim_desc, sim_facts)
            
            if max_sim >= similarity_threshold:
                # Find character positions for this chunk
                chunk_start = text.lower().find(chunk.lower())
                if chunk_start != -1:
                    chunk_end = chunk_start + len(chunk)
                    # Avoid duplicates
                    if not any(start <= chunk_start <= end or start <= chunk_end <= end 
                              for start, end in positions):
                        positions.append((chunk_start, chunk_end))
                        print(f"   Semantic match found (sim={max_sim:.3f}): '{chunk}'")
        
        return positions
    
    def apply_embedding_steering(
        self, 
        text: str, 
        entity_name: str,
        boost_factor: float = 1.0,
        similarity_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies embedding steering using semantic search to find injection positions.

        Args:
            text (str): The full input text.
            entity_name (str): The name of the entity/memory to steer.
            boost_factor (float): Multiplier for memory vector strength.
            similarity_threshold (float): Minimum similarity for semantic injection.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (modified_embeddings, attention_mask)
        """
        if entity_name not in self.entity_memories:
            tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                base_embeddings = self.embedding_layer(tokens['input_ids'])
            return base_embeddings, tokens['attention_mask']

        # Tokenize with offsets
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True, 
            truncation=True
        )
        token_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        offset_mapping = tokens['offset_mapping'][0]
        
        # Find all injection positions
        injection_positions = self._find_memory_injection_positions(
            text, entity_name, similarity_threshold
        )
        
        if not injection_positions:
            print(f"   No injection positions found for '{entity_name}'")
            with torch.no_grad():
                base_embeddings = self.embedding_layer(token_ids)
            return base_embeddings, attention_mask
        
        print(f"   Found {len(injection_positions)} injection position(s)")
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.embedding_layer(token_ids)
        
        modified_embeddings = base_embeddings.clone()
        
        # Get memory embedding
        memory_embedding = self.entity_memories[entity_name]['embedding'].to(self.device)
        scaled_memory = memory_embedding * boost_factor
        
        # Apply injection at each position
        for char_start, char_end in injection_positions:
            # Find tokens that overlap with this character range
            affected_tokens = []
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start < char_end and tok_end > char_start:
                    affected_tokens.append(i)
            
            if not affected_tokens:
                continue
            
            # Primary token gets strongest injection
            primary_idx = affected_tokens[0]
            blend_ratio = 0.8 * boost_factor
            blend_ratio = min(blend_ratio, 0.95)
            
            original = modified_embeddings[0, primary_idx]
            modified_embeddings[0, primary_idx] = (
                (1 - blend_ratio) * original + blend_ratio * scaled_memory
            )
            
            """# Additional tokens in the span get medium injection
            for idx in affected_tokens[1:]:
                original = modified_embeddings[0, idx]
                span_blend = 0.5 * boost_factor
                span_blend = min(span_blend, 0.7)
                modified_embeddings[0, idx] = (
                    (1 - span_blend) * original + span_blend * scaled_memory
                )"""
            
            # Neighboring tokens get light injection
            for offset in [-1, 1]:
                neighbor_idx = primary_idx + offset
                if (0 <= neighbor_idx < modified_embeddings.shape[1] and 
                    neighbor_idx not in affected_tokens):
                    neighbor_original = modified_embeddings[0, neighbor_idx]
                    neighbor_blend = 0.3 * boost_factor
                    neighbor_blend = min(neighbor_blend, 0.5)
                    modified_embeddings[0, neighbor_idx] = (
                        (1 - neighbor_blend) * neighbor_original + neighbor_blend * scaled_memory
                    )
        
        return modified_embeddings, attention_mask
    
    def generate_with_memory(
        self,
        text: str,
        entity_name: str,
        boost_factor: float = 1.0,
        use_context_prefix: bool = True,
        similarity_threshold: float = 0.7,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> str:
        """
        Generates text using the hybrid memory system with semantic injection.

        Args:
            text (str): The user's query.
            entity_name (str): The entity key for memory retrieval.
            boost_factor (float): Strength multiplier for embedding steering.
            use_context_prefix (bool): Whether to prepend stored context.
            similarity_threshold (float): Minimum similarity for semantic injection.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated response.
        """
        # Step 1: Add stored context prefix if enabled
        if use_context_prefix and entity_name in self.entity_memories:
            context_prefix = self.entity_memories[entity_name]['context_prefix']
            modified_text = context_prefix + " " + text
            print(f"\n🔍 Modified prompt: '{modified_text}'")
        else:
            modified_text = text
            print(f"\n🔍 Original prompt: '{text}'")
        
        # Step 2: Apply embedding steering with semantic search
        modified_embeddings, attention_mask = self.apply_embedding_steering(
            modified_text, entity_name, boost_factor, similarity_threshold
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
        bot = BagOfTransformsV5()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 1: Mickey the Cat (Proper Noun Entity)
    print("\n" + "🐱"*40)
    print("TEST 1: PROPER NOUN ENTITY MEMORY")
    print("🐱"*40)
    
    bot.create_entity_memory(
        name="Mickey",
        description="Mickey is a 7 year old cat with black and white fur who likes to eat fish",
        scale=3.5,
        is_proper_noun=True
    )
    
    bot.compare_methods(
        text="Tell me about Mickey",
        entity_name="Mickey",
        max_new_tokens=100
    )
    
    # Test 2: Office Procedure (General Fact Memory)
    print("\n\n" + "🗑️"*40)
    print("TEST 2: PROCEDURAL FACT MEMORY")
    print("🗑️"*40)
    
    bot.create_entity_memory(
        name="trash_procedure",
        description="In the office, when we take out the trash we always need to make sure we use the green bags. Never use blue bags for trash.",
        scale=3.5,
        is_proper_noun=False
    )
    
    bot.compare_methods(
        text="What's the rule about taking out garbage at work?",
        entity_name="trash_procedure",
        max_new_tokens=100
    )
    
    # Test 3: Person Entity with Multiple Attributes
    print("\n\n" + "👤"*40)
    print("TEST 3: PERSON ENTITY WITH MULTIPLE ATTRIBUTES")
    print("👤"*40)
    
    bot.create_entity_memory(
        name="Sarah",
        description="Sarah is a 32 year old software engineer who works at Google and enjoys hiking on weekends",
        scale=3.5,
        is_proper_noun=True
    )
    
    bot.compare_methods(
        text="What do you know about Sarah?",
        entity_name="Sarah",
        max_new_tokens=100
    )
    
    # Test 4: List-based Memory Creation
    print("\n\n" + "🚗"*40)
    print("TEST 4: LIST-BASED MEMORY CREATION")
    print("🚗"*40)
    
    bot.create_entity_memory(
        name="Speedster",
        description=[
            "high-performance sports car",
            "silver and white trim",
            "manufactured in Germany",
            "requires high-octane fuel",
            "top speed 200 mph"
        ],
        scale=3.5,
        is_proper_noun=True
    )
    
    bot.compare_methods(
        text="Tell me about the Speedster vehicle",
        entity_name="Speedster",
        max_new_tokens=100
    )
    
    # Test 5: Semantic Similarity Test (No Exact Name Match)
    print("\n\n" + "🔍"*40)
    print("TEST 5: SEMANTIC SIMILARITY (INDIRECT REFERENCE)")
    print("🔍"*40)
    
    print("\nTesting if semantic search finds Mickey when query uses 'my pet'...")
    result = bot.generate_with_memory(
        text="What does my pet like to eat?",
        entity_name="Mickey",
        boost_factor=2.0,
        use_context_prefix=True,
        similarity_threshold=0.6,  # Lower threshold for this test
        max_new_tokens=80
    )
    print(f"\n📄 OUTPUT:\n\n{result}")
    
    # Test 6: Multiple Memory Injection Points
    print("\n\n" + "💡"*40)
    print("TEST 6: MULTIPLE INJECTION POINTS IN SAME TEXT")
    print("💡"*40)
    
    print("\nTesting multiple mentions of Sarah in one query...")
    result = bot.generate_with_memory(
        text="Sarah is great at her job. I wonder what Sarah does for hobbies? Does Sarah work remotely?",
        entity_name="Sarah",
        boost_factor=1.5,
        use_context_prefix=False,  # Test steering-only
        similarity_threshold=0.7,
        max_new_tokens=100
    )
    print(f"\n📄 OUTPUT:\n\n{result}")
    
    # Test 7: Low Similarity Threshold Test
    print("\n\n" + "⚙️"*40)
    print("TEST 7: VARYING SIMILARITY THRESHOLDS")
    print("⚙️"*40)
    
    for threshold in [0.5, 0.7, 0.9]:
        print(f"\n--- Similarity Threshold: {threshold} ---")
        result = bot.generate_with_memory(
            text="What's the procedure for waste disposal?",
            entity_name="trash_procedure",
            boost_factor=1.5,
            use_context_prefix=False,
            similarity_threshold=threshold,
            max_new_tokens=60
        )
        print(f"OUTPUT: {result[:200]}...")
    
    # Summary Statistics
    print("\n\n" + "📊"*40)
    print("MEMORY SYSTEM STATISTICS")
    print("📊"*40)
    
    print(f"\nTotal memories stored: {len(bot.entity_memories)}")
    for name, memory in bot.entity_memories.items():
        print(f"\n  Memory: '{name}'")
        print(f"    - Embedding norm: {memory['embedding'].norm().item():.2f}")
        print(f"    - Scale factor: {memory['scale']:.1f}x")
        print(f"    - Key facts: {memory['key_facts']}")
        print(f"    - Context prefix: {memory['context_prefix']}")
        print(f"    - Calibrated norm: {memory['avg_token_norm']:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
    
    
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
        text="I've been lurking in this community for a long time, learning so much from all of you, and I'm really grateful. I'm excited to finally be able to contribute something back in case it helps someone else. Where does Sarah work?",
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