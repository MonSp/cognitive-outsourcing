"""Multi-Token Prediction (MTP) Heads for Qwen3.5 Models.

Implements lightweight prediction heads that predict K future tokens
from the model's hidden states, enabling speculative decoding without
requiring a separate drafter model.

Architecture (following DeepSeek-V3):
- Shared projection: hidden_size → hidden_size (optional)
- Per-position heads: hidden_size → vocab_size (K heads)

Training:
- Extract hidden states from Qwen3.5 during normal inference
- Train heads to predict tokens at positions t+1, t+2, ..., t+K
- Use cross-entropy loss

Inference:
- At each generation step, use MTP heads to predict K future tokens
- Verify predictions against the target model
- Accept consecutive matching tokens
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class MTPHead(nn.Module):
    """Single MTP prediction head.

    Takes hidden states and predicts token probabilities at a
    specific offset position (t+1, t+2, ..., t+K).
    """

    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict token logits.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        return self.proj(hidden_states)


class MTPModel(nn.Module):
    """Multi-Token Prediction model with K prediction heads.

    Trains K independent heads, each predicting tokens at a different
    offset position. During inference, all K heads are used for
    speculative decoding.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        dropout: float = 0.1,
        shared_proj: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Optional shared projection before heads
        self.shared_proj = None
        if shared_proj:
            self.shared_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

        # K independent prediction heads
        self.heads = nn.ModuleList([
            MTPHead(hidden_size, vocab_size, dropout)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            targets: List of K tensors, each (batch_size, seq_len) of target token IDs

        Returns:
            Dict with 'logits' (list of K logit tensors) and optionally 'loss'
        """
        if self.shared_proj is not None:
            hidden_states = self.shared_proj(hidden_states)

        logits_list = []
        for head in self.heads:
            logits_list.append(head(hidden_states))

        result = {'logits': logits_list}

        if targets is not None:
            total_loss = 0.0
            loss_list = []
            for k, (logits, target) in enumerate(zip(logits_list, targets)):
                # Flatten for cross-entropy
                logits_flat = logits.reshape(-1, self.vocab_size)
                target_flat = target.reshape(-1)
                loss = F.cross_entropy(logits_flat, target_flat, ignore_index=-100)
                loss_list.append(loss)
                total_loss = total_loss + loss

            result['loss'] = total_loss / len(loss_list)
            result['loss_per_head'] = loss_list

        return result

    def predict(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.0,
    ) -> List[int]:
        """Predict K future tokens from the last hidden state.

        Args:
            hidden_states: (1, seq_len, hidden_size) or (seq_len, hidden_size)
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            List of K predicted token IDs
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        # Convert to float32 for MTP heads
        hidden_states = hidden_states.float()

        # Use only the last position's hidden state
        last_hidden = hidden_states[:, -1:, :]  # (1, 1, hidden_size)

        if self.shared_proj is not None:
            last_hidden = self.shared_proj(last_hidden)

        predictions = []
        for head in self.heads:
            logits = head(last_hidden)  # (1, 1, vocab_size)
            if temperature == 0.0:
                token = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs.squeeze(), 1).item()
            predictions.append(token)

        return predictions


class MTPTrainer:
    """Trains MTP heads on Qwen3.5 hidden states.

    Uses HuggingFace transformers to extract hidden states, then
    trains the MTP heads on a dataset of text sequences.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_heads: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        max_seq_len: int = 512,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer (try ModelScope first, then HuggingFace)
        print(f"Loading model {model_name}...")

        try:
            from modelscope import AutoTokenizer, AutoModelForCausalLM
            print("  Using ModelScope")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
            )
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("  Using HuggingFace Transformers")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        self.model.eval()

        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        print(f"Model loaded: hidden_size={self.hidden_size}, "
              f"vocab_size={self.vocab_size}, num_heads={num_heads}")

    def extract_hidden_states(
        self,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden states and target tokens from texts.

        Args:
            texts: List of text strings

        Returns:
            hidden_states: (total_tokens, hidden_size)
            target_ids: (total_tokens,) of next-token IDs
        """
        all_hidden = []
        all_targets = []

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=self.max_seq_len,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

            # Last hidden state: (1, seq_len, hidden_size)
            hidden = outputs.hidden_states[-1].squeeze(0).float()  # Convert to float32
            input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)

            # Targets are shifted by 1 (next token prediction)
            # For K-head MTP, we need targets at t+1, t+2, ..., t+K
            all_hidden.append(hidden[:-1].cpu())  # Exclude last position
            all_targets.append(input_ids[1:].cpu())  # Exclude first position

        return torch.cat(all_hidden, dim=0), torch.cat(all_targets, dim=0)

    def prepare_mtp_targets(
        self,
        input_ids: torch.Tensor,
        num_heads: int,
        target_len: int,
    ) -> List[torch.Tensor]:
        """Prepare K target tensors for MTP training.

        Args:
            input_ids: (seq_len,) of token IDs
            num_heads: Number of MTP heads (K)
            target_len: Length of target tensor (should match hidden_states length)

        Returns:
            List of K tensors, each (target_len,) with targets at offset positions
        """
        targets = []
        seq_len = len(input_ids)

        for k in range(num_heads):
            # Target at position t is the token at position t+k+1
            target = torch.full((target_len,), -100, dtype=torch.long)
            # Fill in the targets where we have them
            for t in range(target_len):
                src_idx = t + k + 1  # +1 because hidden_states starts from position 0
                if src_idx < seq_len:
                    target[t] = input_ids[src_idx]
            targets.append(target)

        return targets

    def train(
        self,
        texts: List[str],
        num_epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train MTP heads on text data.

        Args:
            texts: List of training text strings
            num_epochs: Number of training epochs
            save_path: Path to save trained heads

        Returns:
            Dict with training history
        """
        # Initialize MTP model
        mtp_model = MTPModel(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            num_heads=self.num_heads,
            shared_proj=True,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            mtp_model.parameters(), lr=self.learning_rate)

        history = {'loss': [], 'loss_per_head': [[] for _ in range(self.num_heads)]}

        print(f"Training MTP heads on {len(texts)} texts for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_head_losses = [0.0] * self.num_heads
            num_batches = 0

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Extract hidden states
                hidden_states, target_ids = self.extract_hidden_states(batch_texts)

                # Prepare MTP targets
                all_targets = []
                for j in range(len(batch_texts)):
                    # Find the start index for this text in the concatenated targets
                    start_idx = 0
                    for prev_j in range(j):
                        prev_inputs = self.tokenizer(
                            batch_texts[prev_j], return_tensors="pt",
                            truncation=True, max_length=self.max_seq_len)
                        start_idx += prev_inputs["input_ids"].shape[1] - 1

                    # Get targets for this text
                    curr_inputs = self.tokenizer(
                        batch_texts[j], return_tensors="pt",
                        truncation=True, max_length=self.max_seq_len)
                    curr_ids = curr_inputs["input_ids"].squeeze(0)
                    # target_len should match hidden_states length (seq_len - 1)
                    text_target_len = curr_ids.shape[0] - 1
                    text_targets = self.prepare_mtp_targets(
                        curr_ids, self.num_heads, text_target_len)

                    if j == 0:
                        all_targets = text_targets
                    else:
                        for k in range(self.num_heads):
                            all_targets[k] = torch.cat(
                                [all_targets[k], text_targets[k]], dim=0)

                # Move to device
                hidden_states = hidden_states.to(self.device)
                all_targets = [t.to(self.device) for t in all_targets]

                # Forward pass
                mtp_model.train()
                result = mtp_model(hidden_states, all_targets)

                loss = result['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                for k, head_loss in enumerate(result['loss_per_head']):
                    epoch_head_losses[k] += head_loss.item()
                num_batches += 1

            # Record history
            avg_loss = epoch_loss / max(1, num_batches)
            history['loss'].append(avg_loss)
            for k in range(self.num_heads):
                avg_head_loss = epoch_head_losses[k] / max(1, num_batches)
                history['loss_per_head'][k].append(avg_head_loss)

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        # Save if requested
        if save_path:
            self.save_mtp_model(mtp_model, save_path, history)

        return history

    def save_mtp_model(
        self,
        mtp_model: MTPModel,
        save_path: str,
        history: Dict,
    ):
        """Save trained MTP model and training history."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Save model weights using safetensors or pickle
        import pickle
        state_dict = mtp_model.state_dict()
        # Convert to CPU tensors for portability
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        with open(os.path.join(save_path, "mtp_heads.pkl"), "wb") as f:
            pickle.dump(cpu_state_dict, f)

        # Save config
        config = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "model_name": self.model_name,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save history
        with open(os.path.join(save_path, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"MTP model saved to {save_path}")

    @staticmethod
    def load_mtp_model(save_path: str, device: str = "cuda") -> MTPModel:
        """Load a trained MTP model."""
        import pickle

        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)

        mtp_model = MTPModel(
            hidden_size=config["hidden_size"],
            vocab_size=config["vocab_size"],
            num_heads=config["num_heads"],
            shared_proj=True,
        )

        # Try loading pickle format first, then torch format
        pkl_path = os.path.join(save_path, "mtp_heads.pkl")
        pt_path = os.path.join(save_path, "mtp_heads.pt")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                state_dict = pickle.load(f)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(f"No model file found in {save_path}")

        mtp_model.load_state_dict(state_dict)
        mtp_model = mtp_model.to(device)
        mtp_model.eval()

        print(f"MTP model loaded from {save_path}: "
              f"num_heads={config['num_heads']}, "
              f"hidden_size={config['hidden_size']}")

        return mtp_model


class MTPSpeculativeDecoder:
    """Speculative decoding using MTP heads.

    One forward pass yields both logits (for verification) and hidden
    states (for MTP drafting). Stop conditions only examine newly
    generated tokens, not the full context.
    """

    def __init__(self, mtp_model, tokenizer, hf_model, device="cuda"):
        self.mtp_model = mtp_model
        self.tokenizer = tokenizer
        self.hf_model = hf_model
        self.device = torch.device(device)
        self.mtp_model.eval()
        self.hf_model.eval()

    def _check_stop(self, new_token_ids):
        if not new_token_ids:
            return False
        text = self.tokenizer.decode(new_token_ids)
        return any(s in text for s in ['\nUser:', '</s>', '</im_end>'])

    def generate(self, input_ids, max_new_tokens=50, current_k=None):
        """Generate with MTP speculative decoding using KV cache.

        Uses past_key_values to avoid reprocessing the full sequence
        during verification. The verification pass only processes K
        draft tokens, not the entire sequence.
        """
        input_ids = input_ids.to(self.device)
        context_len = input_ids.shape[1]
        generated = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        K = current_k or self.mtp_model.num_heads
        tokens_generated = 0
        past_kv = None

        while tokens_generated < max_new_tokens:
            # Step 1: Forward pass (use KV cache if available)
            if past_kv is not None:
                # Only process the last token(s) with cached KV
                new_input = generated[:, -1:]
                outputs = self.hf_model(
                    input_ids=new_input,
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                outputs = self.hf_model(
                    input_ids=generated,
                    output_hidden_states=True,
                    use_cache=True,
                )
            past_kv = outputs.past_key_values
            target_logits = outputs.logits[:, -1, :]
            target_token = target_logits.argmax(dim=-1).item()

            # Step 2: MTP predicts K future tokens
            last_hidden = outputs.hidden_states[-1][:, -1:, :].float()
            draft_tokens = self.mtp_model.predict(last_hidden)[:K]

            # Step 3: Verify drafts using cached KV (only K forward tokens)
            draft_tensor = torch.tensor([draft_tokens], device=self.device)
            verify_outputs = self.hf_model(
                input_ids=draft_tensor,
                past_key_values=past_kv,
                use_cache=True,
            )

            # Step 4: Check each draft token
            accepted = 0
            for i in range(len(draft_tokens)):
                verify_logits = verify_outputs.logits[:, i, :]
                expected_token = verify_logits.argmax(dim=-1).item()
                if draft_tokens[i] == expected_token:
                    accepted += 1
                else:
                    break

            # Step 5: Update sequence and KV cache
            if accepted > 0:
                kept = draft_tensor[:, :accepted]
                generated = torch.cat([generated, kept], dim=1)
                tokens_generated += accepted
                # Update past_kv to include accepted tokens
                # The verify_outputs already has KV for all K tokens
                # We need to trim to only accepted tokens
                past_kv = verify_outputs.past_key_values
            else:
                target_tensor = torch.tensor([[target_token]], device=self.device)
                generated = torch.cat([generated, target_tensor], dim=1)
                tokens_generated += 1
                # Use the verify pass KV (it processed the target token position)
                past_kv = verify_outputs.past_key_values

            # Check stop conditions
            new_token_ids = generated[0, context_len:].tolist()
            if self._check_stop(new_token_ids):
                break
            if eos_token_id is not None and eos_token_id in new_token_ids:
                break

        return generated

    def generate_with_sig(self, context_ids, max_new_tokens=50,
                          warmup_steps=3, max_k=None):
        """Generate with SIG-aware adaptive K scheduling.

        Each step: 1 full forward pass + 1 verify pass on extended sequence.
        Accept consecutive matching draft tokens.
        """
        if max_k is None:
            max_k = self.mtp_model.num_heads
        context_ids = context_ids.to(self.device)
        context_len = context_ids.shape[1]
        generated = context_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        current_k = 1
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            if tokens_generated >= warmup_steps and current_k < max_k:
                current_k = min(current_k + 1, max_k)

            # Full forward pass
            outputs = self.hf_model(
                input_ids=generated,
                output_hidden_states=True,
            )
            target_logits = outputs.logits[:, -1, :]
            target_token = target_logits.argmax(dim=-1).item()
            last_hidden = outputs.hidden_states[-1][:, -1:, :].float()
            draft_tokens = self.mtp_model.predict(last_hidden)[:current_k]

            # Verify: extend sequence with drafts and check logits
            draft_tensor = torch.tensor([draft_tokens], device=self.device)
            extended = torch.cat([generated, draft_tensor], dim=1)
            verify_outputs = self.hf_model(input_ids=extended)

            accepted = 0
            for i in range(len(draft_tokens)):
                pos = generated.shape[1] + i
                if pos >= verify_outputs.logits.shape[1]:
                    break
                verify_logits = verify_outputs.logits[:, pos - 1, :]
                expected_token = verify_logits.argmax(dim=-1).item()
                if draft_tokens[i] == expected_token:
                    accepted += 1
                else:
                    break

            if accepted > 0:
                kept = draft_tensor[:, :accepted]
                generated = torch.cat([generated, kept], dim=1)
                tokens_generated += accepted
            else:
                target_tensor = torch.tensor([[target_token]], device=self.device)
                generated = torch.cat([generated, target_tensor], dim=1)
                tokens_generated += 1

            new_token_ids = generated[0, context_len:].tolist()
            if self._check_stop(new_token_ids):
                break
            if eos_token_id is not None and eos_token_id in new_token_ids:
                break

        return generated


def prepare_training_data(
    model_path: str = "models/Qwen3.5-4B-Q4_K_M.gguf",
    num_samples: int = 100,
    max_seq_len: int = 512,
) -> List[str]:
    """Prepare training data from EdgeAgent-Kitchen scenarios.

    Uses the kitchen tool registry to generate diverse training texts
    that cover the domain of the benchmark.
    """
    import sys
    import random
    sys.path.insert(0, '.')
    from edge_agent_bench import KitchenToolRegistry, build_kitchen_scenario

    tools = KitchenToolRegistry()
    texts = []

    # Generate scenarios with different seeds
    for seed in range(num_samples):
        random.seed(seed)
        scenario = build_kitchen_scenario(min(20, 50))  # Shorter for training

        context = "You are an intelligent kitchen assistant.\n\n"
        for step in scenario:
            context += f"User: {step.user_query}\n"
            result = tools.execute(step.tool_name, step.tool_args)
            context += f"[Tool: {step.tool_name}] {result}\nAssistant: "

            # Generate a response (use tool info to construct)
            if step.tool_name == "check_pantry":
                context += "I've checked the pantry. You have various ingredients available.\n\n"
            elif step.tool_name == "get_recipe":
                context += f"Here's the recipe for {step.tool_args.get('recipe_id', 'the dish')}.\n\n"
            elif step.tool_name == "set_oven":
                context += f"Oven set to {step.tool_args.get('temp_c', 180)}°C.\n\n"
            elif step.tool_name == "check_ingredients":
                context += "Let me check what ingredients you have for this recipe.\n\n"
            else:
                context += "I'll help you with that.\n\n"

            # Truncate to max_seq_len
            if len(context) > max_seq_len * 4:  # Approximate token count
                texts.append(context[:max_seq_len * 4])
                context = context[-max_seq_len * 2:]  # Keep some overlap

    texts.append(context)  # Add final context
    return texts[:num_samples]
