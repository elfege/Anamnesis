"""
inference.py — Generation with uncertainty detection and bassin recall.

This is where the bassin de tenseurs potentiels gets USED at inference time.

Standard inference:
    prompt → model → tokens → done

δ² inference:
    prompt → model → tokens → CHECK CONFIDENCE
        if confident → done
        if uncertain → QUERY BASSIN → inject contrastive context → re-generate

The uncertainty trigger is ENTROPY of the output distribution:
    Low entropy  = model is confident (one token dominates)
    High entropy = model is uncertain (probability spread across many tokens)

When uncertain, we look at the bassin for past tensions that are
semantically related to the current context. These tensions tell us:
"When you were training, you had strong opposing forces here. Consider
both sides."

This is Plato's anamnesis: learning is recollection. The bassin holds
what was "forgotten" (discarded by standard training) and retrieves it
at the moment of doubt.
"""

import json
import logging
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch
from torch.nn import functional as F

from neural_network import Transformer, TransformerConfig
from bassin import compute_entropy

logger = logging.getLogger("anamnesis.d2.inference")


class D2InferenceEngine:
    """
    Inference engine for δ²-trained models.

    Loads a trained checkpoint, generates text, and optionally triggers
    bassin recall when the model is uncertain.

    Usage:
        engine = D2InferenceEngine(checkpoint_path="d2/output/best.pt")
        engine.load()
        text = engine.generate("What is dialectics?", max_tokens=200)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        entropy_threshold: float = 4.0,
        enable_bassin_recall: bool = True,
    ):
        """
        Args:
            checkpoint_path:   path to a .pt checkpoint file
            device:            "auto", "cuda", or "cpu"
            entropy_threshold: entropy above this triggers bassin recall
                               (typical range: 3.0-5.0; uniform over 50k ≈ 10.8)
            enable_bassin_recall: whether to query bassin on uncertainty
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.entropy_threshold = entropy_threshold
        self.enable_bassin_recall = enable_bassin_recall

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[Transformer] = None
        self.config: Optional[dict] = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """
        Load the model from checkpoint.

        Returns:
            True on success, False on failure.
        """
        with self._lock:
            if self.model is not None:
                return True

            if not self.checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {self.checkpoint_path}")
                return False

            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )

                # Rebuild the model config from the checkpoint
                cfg = checkpoint.get('config', {})
                model_config = TransformerConfig(
                    block_size=cfg.get('block_size', 256),
                    vocab_size=cfg.get('vocab_size', 50304),
                    n_layer=cfg.get('n_layer', 6),
                    n_head=cfg.get('n_head', 6),
                    n_embd=cfg.get('n_embd', 384),
                    dropout=0.0,  # no dropout at inference
                    bias=cfg.get('bias', False),
                )

                self.model = Transformer(model_config).to(self.device)
                self.model.load_state_dict(checkpoint['model'])
                self.model.eval()
                self.config = cfg

                logger.info(
                    f"Model loaded: {model_config.n_layer}L/{model_config.n_head}H/"
                    f"{model_config.n_embd}E, "
                    f"step={checkpoint.get('step', '?')}, "
                    f"val_loss={checkpoint.get('val_loss', '?')}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def is_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 200,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate tokens with optional uncertainty detection.

        Args:
            prompt_ids:  (1, T) tensor of token IDs (the prompt, already tokenized)
            max_tokens:  how many tokens to generate
            temperature: randomness control
            top_k:       only sample from top K tokens

        Returns:
            (generated_ids, stats)
            generated_ids: (1, T + max_tokens) tensor
            stats: dict with generation metadata including entropy measurements
                   and any bassin recall events
        """
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Model not loaded")

        idx = prompt_ids.to(self.device)
        stats = {
            "tokens_generated": 0,
            "entropy_measurements": [],
            "bassin_recalls": 0,
            "high_entropy_positions": [],
        }

        for i in range(max_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.model.config.block_size \
                else idx[:, -self.model.config.block_size:]

            # Forward pass
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]  # (1, vocab_size) — last position only

            # ── Uncertainty detection ────────────────────────────────
            entropy = compute_entropy(logits[0])
            stats["entropy_measurements"].append(entropy)

            if entropy > self.entropy_threshold and self.enable_bassin_recall:
                # The model is uncertain. In a full implementation, we would:
                # 1. Query the bassin for semantically related tensions
                # 2. Use those tensions to modify the logits or re-prompt
                # 3. Generate again with the contrastive context
                #
                # For now, we just LOG it. The bassin recall mechanism
                # requires the bassin to be populated (from training with δ²)
                # and a semantic similarity query (using embeddings).
                #
                # TODO: Implement full bassin recall pipeline
                stats["bassin_recalls"] += 1
                stats["high_entropy_positions"].append(i)
                logger.debug(
                    f"Token {i}: entropy={entropy:.2f} > threshold={self.entropy_threshold} "
                    f"— bassin recall triggered (not yet implemented)"
                )

            # ── Sampling ─────────────────────────────────────────────
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            stats["tokens_generated"] += 1

        # Summary stats
        entropies = stats["entropy_measurements"]
        if entropies:
            stats["entropy_mean"] = sum(entropies) / len(entropies)
            stats["entropy_max"] = max(entropies)
            stats["entropy_min"] = min(entropies)
            stats["pct_uncertain"] = len(stats["high_entropy_positions"]) / len(entropies) * 100

        return idx, stats

    def get_status(self) -> dict:
        """Status dict for API endpoints."""
        return {
            "loaded": self.is_loaded(),
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "entropy_threshold": self.entropy_threshold,
            "bassin_recall_enabled": self.enable_bassin_recall,
            "config": self.config,
        }
