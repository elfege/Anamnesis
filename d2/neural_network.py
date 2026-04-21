"""
neural_network.py — The transformer architecture for Anamnesis-δ².

This is the "body" of the neural network. It defines HOW data flows through
the model (forward pass), but NOT how the model learns (that's the optimizer's
job — see optimizer.py).

Adapted from Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT),
heavily commented for understanding. Every line is explained.

If you're reading this for the first time, read top to bottom. The building
blocks are defined first, then assembled into the full model at the end.


THE BIG PICTURE — What this file builds:
=========================================

    "Hello world" (text)
         |
         v
    [Tokenizer]  ──── turns text into numbers (token IDs)
         |               e.g. "Hello" -> [15496], "world" -> [995]
         v
    [Embedding]  ──── turns each token ID into a vector of 768 numbers
         |               e.g. 15496 -> [0.23, -0.87, 0.04, ...]  (768 values)
         |               WHY? Because math works on numbers, not words.
         |               Each dimension captures some aspect of meaning.
         v
    [+ Position] ──── adds "where am I in the sentence?" information
         |               Without this, the model can't tell word order.
         |               "dog bites man" would equal "man bites dog".
         v
    ┌─────────────────────────────────────────┐
    │  Transformer Block (repeated N times)    │
    │                                          │
    │  1. LayerNorm ── stabilize the numbers   │
    │  2. Attention ── "which words matter      │
    │                   for understanding       │
    │                   THIS word?"              │
    │  3. Residual  ── add the original back    │
    │                   (don't lose info)        │
    │  4. LayerNorm ── stabilize again          │
    │  5. FFN       ── "think about it"         │
    │                   (expand, transform,      │
    │                    compress)               │
    │  6. Residual  ── add original back again  │
    │                                          │
    └─────────────────────────────────────────┘
         |
         v  (repeated 12 times for GPT-2 small)
         |
    [Final LayerNorm]
         |
         v
    [Output Layer]  ──── turns vector back into vocabulary probabilities
         |               "given everything so far, what word comes next?"
         |               e.g. [0.001, 0.0003, ..., 0.12, ...] (50,000+ entries)
         v
    "the" (predicted next word)


VOCABULARY FOR THIS FILE:
==========================

    tensor      A multi-dimensional array of numbers. A scalar is 0D, a vector
                is 1D, a matrix is 2D, and beyond that it's just "a tensor."
                PyTorch calls everything a tensor.

    nn.Module   PyTorch's base class for any neural network component. Every
                layer, block, and the full model inherits from this. It gives
                you .parameters() (all learnable weights) and .forward()
                (how data flows through).

    forward()   THE most important method. It defines what happens when data
                enters this component. PyTorch calls it automatically when you
                write: output = my_layer(input)

    nn.Linear   A single matrix multiplication + optional bias addition.
                output = input @ weight.T + bias
                This is THE fundamental operation. Everything else is just
                clever arrangements of these.

    nn.Embedding  A lookup table. Row i contains the vector for token i.
                  NOT a computation — just a table lookup. But the table
                  entries ARE learnable weights that change during training.

    B, T, C     The three dimensions you'll see everywhere:
                B = batch size (how many sequences we process in parallel)
                T = time/sequence length (how many tokens in each sequence)
                C = channels/embedding dimension (how many numbers per token)

    logits      The raw output numbers before converting to probabilities.
                Named after the logistic function. Just means "unnormalized
                scores" — higher = model thinks this token is more likely.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# BUILDING BLOCK 1: Layer Normalization
# ============================================================================
#
# Problem: As data flows through many layers, the numbers can drift —
# getting very large or very small. This makes training unstable.
#
# Solution: LayerNorm forces each layer's output to have mean=0 and
# variance=1, then applies a learnable scale (weight) and shift (bias).
#
# Analogy: Like re-centering and re-scaling a photograph's brightness
# and contrast at each step of processing, so it doesn't gradually
# wash out or go black.
#
# Math:
#   output = (input - mean) / sqrt(variance + epsilon) * weight + bias
#
#   epsilon (1e-5 = 0.00001) prevents division by zero.
#   weight and bias are LEARNED — the model decides what scale/shift is best.
# ============================================================================

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        """
        Args:
            ndim: number of features to normalize (= embedding dimension C)
            bias: whether to include a learnable bias term
        """
        super().__init__()
        # weight starts at 1.0 (no scaling initially)
        self.weight = nn.Parameter(torch.ones(ndim))
        # bias starts at 0.0 (no shifting initially), or None if disabled
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # F.layer_norm does the math described above.
        # self.weight.shape tells it which dimensions to normalize over.
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ============================================================================
# BUILDING BLOCK 2: Causal Self-Attention
# ============================================================================
#
# This is THE core mechanism of the transformer. Everything else is just
# bookkeeping around this.
#
# THE QUESTION ATTENTION ANSWERS:
#   "For each word in the sequence, which OTHER words should I pay
#    attention to in order to understand THIS word?"
#
# Example: "The cat sat on the mat because it was tired"
#   When processing "it", attention figures out that "it" refers to "cat"
#   (not "mat") by computing how strongly "it" should attend to each
#   previous word.
#
# HOW IT WORKS — The Q/K/V mechanism:
#
#   For each token, we compute three vectors:
#
#   Q (Query)  = "What am I looking for?"
#                The token asks a question about what information it needs.
#
#   K (Key)    = "What do I contain?"
#                Each token advertises what information it has.
#
#   V (Value)  = "What information do I actually give you?"
#                The actual content that gets passed along.
#
#   Attention score = Q · K^T  (dot product: how well does my query
#                                match your key?)
#
#   High score = "this token has what I'm looking for" → attend strongly
#   Low score  = "this token is irrelevant to me" → ignore
#
#   Final output = weighted sum of V vectors, weighted by attention scores.
#
# CAUSAL means: each token can only attend to tokens BEFORE it (and itself).
# Token 5 can look at tokens 0-5 but NOT tokens 6+. This is what makes it
# a LANGUAGE MODEL — it predicts the future from the past, never peeks ahead.
#
# MULTI-HEAD means: we do this attention computation multiple times in
# parallel (e.g., 12 times for GPT-2 small), each with different learned
# Q/K/V matrices. Each "head" can learn to attend to different things:
#   - Head 1 might learn syntactic relationships (subject-verb)
#   - Head 2 might learn semantic relationships (pronoun-antecedent)
#   - Head 3 might learn positional patterns (nearby words)
#   etc.
#
# Philosophically (per Elfege's Hegelian reading): attention is the mechanism
# by which each token predicates on every other token — "being-for-other."
# The token does not exist in monadic isolation; it exists through its
# relations to all other tokens in the sequence.
# ============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # The embedding dimension must be evenly divisible by the number
        # of attention heads. Each head gets (n_embd / n_head) dimensions.
        # Example: 768 dims / 12 heads = 64 dims per head.
        assert config.n_embd % config.n_head == 0

        # This single linear layer computes Q, K, and V all at once.
        # Input:  (B, T, C)        e.g. (batch, 1024 tokens, 768 dims)
        # Output: (B, T, 3 * C)    e.g. (batch, 1024 tokens, 2304 dims)
        #                          first 768 = Q, next 768 = K, last 768 = V
        # WHY all at once? Efficiency. One big matrix multiply is faster
        # than three separate ones on a GPU.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # After attention, project back to the embedding dimension.
        # This combines the outputs of all heads into a single vector.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout: randomly zero out some values during training.
        # This prevents the model from relying too heavily on any single
        # attention pattern. Like training with random blindfolds — forces
        # the model to develop redundant, robust representations.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash Attention is a GPU-optimized version of the same math.
        # Produces identical results, just faster. Available in PyTorch >= 2.0.
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # If flash isn't available, we need a causal mask.
            # This is a lower-triangular matrix of 1s:
            #   [[1, 0, 0, 0],
            #    [1, 1, 0, 0],
            #    [1, 1, 1, 0],
            #    [1, 1, 1, 1]]
            # The 0s become -infinity after masking, which become 0 after
            # softmax. This enforces: "token i can only attend to tokens <= i."
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        # x shape: (B, T, C) = (batch_size, sequence_length, embedding_dim)
        B, T, C = x.size()

        # ── Step 1: Compute Q, K, V ──────────────────────────────────────
        # One big matrix multiply, then split into three equal parts.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Each is now (B, T, C) = (batch, seq_len, 768)

        # ── Step 2: Reshape for multi-head attention ─────────────────────
        # We split the 768 dimensions into 12 heads of 64 dims each.
        # Before: (B, T, C)        = (batch, 1024, 768)
        # After:  (B, n_head, T, head_size) = (batch, 12, 1024, 64)
        #
        # .view() reshapes: (B, T, 768) -> (B, T, 12, 64)
        # .transpose(1,2) swaps dims 1 and 2: (B, T, 12, 64) -> (B, 12, T, 64)
        # This puts the head dimension in position 1, so each head is
        # processed independently (like a batch dimension).
        head_size = C // self.n_head  # 768 / 12 = 64
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, 12, T, 64)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, 12, T, 64)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, 12, T, 64)

        # ── Step 3: Compute attention scores and apply ───────────────────
        if self.flash:
            # Flash Attention: same math as below, but fused into one GPU
            # kernel for ~2-4x speedup. is_causal=True handles the mask.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention — shown here for understanding.
            #
            # Step 3a: Score = Q · K^T / sqrt(head_size)
            #   (B, 12, T, 64) @ (B, 12, 64, T) -> (B, 12, T, T)
            #   Each entry [i][j] = "how much should token i attend to token j?"
            #
            #   Division by sqrt(64) = 8 prevents the dot products from getting
            #   too large, which would make softmax output near-0 or near-1
            #   (saturated gradients = can't learn).
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Step 3b: Apply causal mask — set future positions to -infinity.
            #   After softmax, -inf becomes 0 = zero attention to future tokens.
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            # Step 3c: Softmax — convert scores to probabilities (sum to 1).
            #   Now each row is a probability distribution: "how much should
            #   token i attend to each of tokens 0..i?"
            att = F.softmax(att, dim=-1)

            # Step 3d: Dropout on attention weights (during training only).
            att = self.attn_dropout(att)

            # Step 3e: Weighted sum of values.
            #   (B, 12, T, T) @ (B, 12, T, 64) -> (B, 12, T, 64)
            #   Each token's output = sum of all value vectors, weighted by
            #   how much attention it paid to each one.
            y = att @ v

        # ── Step 4: Reassemble heads ─────────────────────────────────────
        # Reverse the reshape: (B, 12, T, 64) -> (B, T, 12, 64) -> (B, T, 768)
        # .contiguous() ensures memory layout is correct for .view()
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ── Step 5: Output projection ────────────────────────────────────
        # One more linear layer to mix the head outputs together.
        # Then dropout for regularization.
        y = self.resid_dropout(self.c_proj(y))
        return y


# ============================================================================
# BUILDING BLOCK 3: Feed-Forward Network (MLP)
# ============================================================================
#
# After attention has gathered information from other tokens, the FFN
# processes that information for each token independently.
#
# If attention is "gathering evidence," FFN is "thinking about it."
#
# Structure:
#   1. Expand: project from 768 dims to 3072 dims (4x wider)
#   2. Activate: apply GELU non-linearity (introduces non-linear "thinking")
#   3. Compress: project back from 3072 dims to 768 dims
#   4. Dropout: regularization
#
# WHY expand then compress?
#   The expansion gives the network a higher-dimensional space to work in —
#   more room to represent complex relationships. Then it compresses back
#   to the original size so the next layer gets the same shape.
#   Like sketching on a large whiteboard, then summarizing on a sticky note.
#
# WHY GELU (Gaussian Error Linear Unit)?
#   Without a non-linearity, stacking linear layers is mathematically
#   equivalent to a single linear layer — you gain nothing from depth.
#   GELU introduces non-linearity: some inputs get amplified, others
#   get dampened. This is what makes neural networks capable of learning
#   complex patterns rather than just straight lines.
#   GELU is smoother than ReLU (the older standard). Smooth = better
#   gradients = easier to train.
# ============================================================================

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Expand: 768 -> 3072
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Non-linearity
        self.gelu = nn.GELU()
        # Compress: 3072 -> 768
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)       # (B, T, 768) -> (B, T, 3072)  expand
        x = self.gelu(x)       # (B, T, 3072) -> (B, T, 3072)  non-linearity
        x = self.c_proj(x)     # (B, T, 3072) -> (B, T, 768)   compress
        x = self.dropout(x)    # (B, T, 768) -> (B, T, 768)    regularize
        return x


# ============================================================================
# BUILDING BLOCK 4: Transformer Block
# ============================================================================
#
# One transformer block = one "layer" of the model.
# GPT-2 small has 12 of these stacked on top of each other.
#
# Each block does two things:
#   1. Attention: gather information from other tokens
#   2. FFN: process/transform that information
#
# Both use RESIDUAL CONNECTIONS (the "+ x" part):
#   output = x + attention(layernorm(x))
#   output = x + ffn(layernorm(x))
#
# WHY residual connections?
#   Without them, gradients have to flow through EVERY layer to reach
#   the early layers during training. With 12+ layers, the gradient
#   signal gets weaker and weaker (vanishing gradient problem).
#
#   The residual connection creates a "highway" — the gradient can flow
#   directly through the addition, bypassing the layer entirely if needed.
#   This makes deep networks trainable.
#
#   Intuition: the layer only needs to learn the DIFFERENCE from its input,
#   not the entire output. Much easier to learn "add a small correction"
#   than "reproduce the entire input plus a correction."
# ============================================================================

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # normalize before attention
        self.attn = CausalSelfAttention(config)                  # attend to other tokens
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # normalize before FFN
        self.mlp = MLP(config)                                   # process/transform

    def forward(self, x):
        # Note the pattern: x = x + layer(norm(x))
        # 1. Normalize x
        # 2. Run through the layer (attention or FFN)
        # 3. ADD the result back to the original x (residual connection)
        x = x + self.attn(self.ln_1(x))   # attend, then add residual
        x = x + self.mlp(self.ln_2(x))    # think, then add residual
        return x


# ============================================================================
# CONFIGURATION
# ============================================================================
#
# All the hyperparameters that define the model's shape.
# Change these = change the model size.
#
# GPT-2 sizes for reference:
#   small:   n_layer=12, n_head=12, n_embd=768   →  124M params
#   medium:  n_layer=24, n_head=16, n_embd=1024  →  350M params
#   large:   n_layer=36, n_head=20, n_embd=1280  →  774M params
#   xl:      n_layer=48, n_head=25, n_embd=1600  → 1558M params
#
# For δ² experiments, we'll use small or even smaller to iterate fast.
# The optimizer is the novel part, not the architecture.
# ============================================================================

@dataclass
class TransformerConfig:
    block_size: int = 1024      # maximum sequence length (how many tokens the model can "see" at once)
    vocab_size: int = 50304     # number of unique tokens (GPT-2 uses 50257, padded to multiple of 64 for GPU efficiency)
    n_layer: int = 12           # number of transformer blocks stacked (depth = capacity for complex reasoning)
    n_head: int = 12            # number of attention heads per block (parallel "perspectives")
    n_embd: int = 768           # embedding dimension (size of each token's vector representation)
    dropout: float = 0.0        # dropout rate: 0.0 = no dropout. Set to 0.1-0.2 during training for regularization
    bias: bool = True           # include bias terms in linear layers. False = slightly faster, slightly better


# ============================================================================
# THE FULL MODEL: Transformer
# ============================================================================
#
# This class assembles all the building blocks into a complete language model.
#
# The model does ONE thing: given a sequence of token IDs, predict the
# probability of the NEXT token.
#
#   Input:  [15496, 995, 318, ...]     (token IDs for "Hello world is ...")
#   Output: [0.001, 0.003, ..., 0.12, ...]  (probability for each possible next token)
#
# During training, we also compute the LOSS: how far off were our predictions
# from the actual next tokens? This loss is what the optimizer (Adam or δ²)
# uses to update the weights.
#
# The architecture looks like this (matching the ASCII diagram at the top):
#
#   wte  = Word Token Embedding     (lookup table: token ID → vector)
#   wpe  = Word Position Embedding  (lookup table: position → vector)
#   drop = Dropout                  (regularization)
#   h    = list of N Blocks         (the transformer layers)
#   ln_f = Final LayerNorm          (stabilize before output)
#   lm_head = Language Model Head   (vector → vocabulary logits)
#
# WEIGHT TYING: wte and lm_head share the same weight matrix.
#   Why? The embedding matrix maps token → vector. The output matrix maps
#   vector → token. These are conceptually inverse operations, so sharing
#   weights makes them consistent AND cuts the parameter count significantly.
# ============================================================================

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ── Build the model components ───────────────────────────────────
        self.transformer = nn.ModuleDict(dict(
            # Token embedding: vocab_size x n_embd matrix.
            # Row i = the vector representation of token i.
            # This is a LOOKUP TABLE with learnable entries.
            wte=nn.Embedding(config.vocab_size, config.n_embd),

            # Position embedding: block_size x n_embd matrix.
            # Row i = "what it means to be in position i."
            # Learned during training — the model figures out what
            # positional information is useful.
            wpe=nn.Embedding(config.block_size, config.n_embd),

            # Dropout after embeddings.
            drop=nn.Dropout(config.dropout),

            # The transformer blocks — this is where the real work happens.
            # A list of N identical blocks, each with attention + FFN.
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer normalization before the output.
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # The "language model head" — projects from embedding space back
        # to vocabulary space. Output is (B, T, vocab_size) = a probability
        # distribution over all possible next tokens.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output projection shares weights with the
        # token embedding. This means the model uses the SAME matrix to
        # go from token→vector and from vector→token.
        self.transformer.wte.weight = self.lm_head.weight

        # ── Initialize all weights ───────────────────────────────────────
        # Neural networks start with random weights. The initialization
        # strategy matters: too large = unstable, too small = vanishing signals.
        # We use normal distribution with std=0.02 (standard for transformers).
        self.apply(self._init_weights)

        # Special initialization for output projections in residual paths.
        # These get scaled down by 1/sqrt(2*n_layer) so that the residual
        # stream doesn't grow too large as we stack more layers.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameter count
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Count the total number of learnable parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Don't count position embeddings — they're not "real" parameters
            # in the traditional sense. Token embeddings ARE counted because
            # they're shared with the output layer (weight tying).
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights for a single module.

        Called by self.apply() which walks through every sub-module.

        Linear layers: normal distribution, mean=0, std=0.02
        Embedding layers: same
        Biases: all zeros
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        THE FORWARD PASS — this is what happens when data flows through the model.

        Args:
            idx:     (B, T) tensor of token IDs. Each row is a sequence.
                     e.g. [[15496, 995, 318, ...], [464, 2563, 286, ...]]
                     B = batch size, T = sequence length

            targets: (B, T) tensor of target token IDs (what SHOULD come next).
                     Only provided during training. During inference, this is None.
                     targets[b][t] = the correct next token after idx[b][t]

        Returns:
            logits:  (B, T, vocab_size) tensor — raw scores for each possible next token
            loss:    scalar tensor — how wrong we were (only if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Create position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # ── Step 1: Embeddings ───────────────────────────────────────────
        # Look up the vector for each token ID
        tok_emb = self.transformer.wte(idx)    # (B, T) -> (B, T, 768)
        # Look up the vector for each position
        pos_emb = self.transformer.wpe(pos)    # (T,) -> (T, 768)
        # Add them together: token meaning + positional information
        # Broadcasting: pos_emb (T, 768) gets added to each batch element
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, 768)

        # ── Step 2: Pass through all transformer blocks ──────────────────
        # Each block refines the representations. Early blocks tend to
        # learn syntax; later blocks tend to learn semantics.
        for block in self.transformer.h:
            x = block(x)  # (B, T, 768) -> (B, T, 768), same shape, different values

        # ── Step 3: Final normalization ──────────────────────────────────
        x = self.transformer.ln_f(x)  # (B, T, 768)

        # ── Step 4: Output ───────────────────────────────────────────────
        if targets is not None:
            # TRAINING: compute logits for ALL positions (need them for loss)
            logits = self.lm_head(x)  # (B, T, 768) -> (B, T, vocab_size)

            # Cross-entropy loss: "how surprised are we by the actual next token?"
            # Reshapes to 2D because cross_entropy expects (N, C) not (B, T, C).
            # ignore_index=-1 means: skip positions where target is -1 (padding).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
                ignore_index=-1,
            )
        else:
            # INFERENCE: only compute logits for the LAST position
            # (we only need to predict the next token, not re-predict all of them)
            # Using [-1] as a list preserves the time dimension for shape consistency.
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    # ========================================================================
    # TEXT GENERATION
    # ========================================================================
    #
    # This is how the model produces text. It's called "autoregressive"
    # generation: predict one token, append it, predict the next, repeat.
    #
    # The model NEVER "thinks ahead" — it only ever predicts one token at
    # a time, based on everything before it.
    #
    # Temperature and top_k control the randomness:
    #   temperature = 1.0: normal randomness
    #   temperature < 1.0: more deterministic (picks the most likely token)
    #   temperature > 1.0: more random (more creative / more chaotic)
    #   top_k = 50: only consider the top 50 most likely tokens (ignore the rest)
    # ========================================================================

    @torch.no_grad()  # don't track gradients during generation (saves memory)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text token by token.

        Args:
            idx:             (B, T) starting token IDs (the prompt)
            max_new_tokens:  how many tokens to generate
            temperature:     randomness control (see above)
            top_k:           only sample from top K most likely tokens

        Returns:
            (B, T + max_new_tokens) — the original prompt + generated tokens
        """
        for _ in range(max_new_tokens):
            # If the sequence is longer than the model can handle, crop it.
            # The model can only "see" block_size tokens at a time.
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]

            # Forward pass: get logits for the next token
            logits, _ = self(idx_cond)

            # Take only the logits for the LAST position (the prediction)
            # and scale by temperature
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # If using top_k: zero out everything below the k-th highest score
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size), sums to 1

            # Sample one token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
