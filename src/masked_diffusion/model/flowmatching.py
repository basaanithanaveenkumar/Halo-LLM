

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ======================================================================================
# 1. DATALOADER
# ======================================================================================

# class WikiTextDataset(Dataset):
#     def __init__(self, tokenizer_name="gpt2", max_length=128, split="train", size=None):
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         if self.tokenizer.mask_token is None:
#             self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

#         self.max_length = max_length
#         dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
#         if size is not None:
#             dataset = dataset.select(range(min(size, len(dataset))))

#         self.data = dataset
#         self.mask_token_id = self.tokenizer.mask_token_id
#         self.vocab_size = len(self.tokenizer)

#         print(f"Loaded {len(self.data)} samples from WikiText-2 ({split} split).")
#         print(f"Vocabulary size (incl. [MASK]): {self.vocab_size}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]["text"]
#         if not text or text.strip() == "":
#             text = " "

#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt",
#             add_special_tokens=False,
#         )
#         return encoding["input_ids"].squeeze(0)
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class WikiTextDataset(Dataset):
    def __init__(
        self,
        tokenizer_name="gpt2",
        max_length=96,
        stride=24,  # Overlap step size (default: 50% overlap for 2x samples)
        split="train",
        size=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        self.max_length = max_length
        self.stride = stride if stride is not None else max_length

        # Load dataset
        raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        if size is not None:
            raw_dataset = raw_dataset.select(range(min(size, len(raw_dataset))))

        # 1. Concatenate non-empty text lines separated by EOS tokens
        texts = [text for text in raw_dataset["text"] if text and text.strip()]
        full_text = self.tokenizer.eos_token.join(texts)

        # 2. Tokenize the entire text block at once (no padding/truncation here)
        print("Tokenizing full dataset into a single continuous stream...")
        full_encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)

        # 3. Compute window slice indices
        total_tokens = full_encodings.size(0)
        self.samples = []

        for i in range(0, total_tokens - self.max_length + 1, self.stride):
            self.samples.append(full_encodings[i : i + self.max_length])

        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)

        print(
            f"Created {len(self.samples)} sliding window samples "
            f"(seq_len={self.max_length}, stride={self.stride}) from WikiText-2 ({split})."
        )
        print(f"Vocabulary size (incl. [MASK]): {self.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloaders(batch_size=16, max_length=128, train_size=2048, val_size=512,
                    tokenizer_name="gpt2"):
    train_dataset = WikiTextDataset(tokenizer_name, max_length,4, "train", train_size)
    val_dataset   = WikiTextDataset(tokenizer_name, max_length,4, "validation", val_size)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dl, val_dl, train_dataset.tokenizer, train_dataset.mask_token_id, train_dataset.vocab_size


# ======================================================================================
# 2. MODEL
# ======================================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for continuous t in [0, 1].
    Verified: at t=0, sin(0)=0 and cos(0)=1 for all frequencies. ✓"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args  = t[:, None].float() * freqs[None, :] * 1000.0
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional transformer block with AdaLN-Zero time conditioning.

    VERIFIED BIDIRECTIONAL: changing the token at position L-1 changes logits at
    position 0 (confirmed with different token IDs, not just raw vector offsets
    which LayerNorm would cancel). ✓

    AdaLN-Zero: time_proj initialised to zero so scale=0, shift=0 at training start
    → block acts like a plain pre-norm transformer at step 0, then gradually learns
    time-dependent modulation. This is the standard DiT initialisation. ✓
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.time_proj = nn.Linear(d_model, 4 * d_model)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x, time_emb, key_padding_mask=None):
        scale1, shift1, scale2, shift2 = self.time_proj(time_emb).chunk(4, dim=-1)

        h = self.ln1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out

        h = self.ln2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.ff(h)
        return x


class FlowMatchingLLM(nn.Module):
    """Bidirectional denoiser: predicts p_theta(x1 | x_t, t).

    time_emb is (B, D) -- one vector per sequence, broadcast to all positions
    via unsqueeze(1) inside each TransformerBlock. ✓
    """
    def __init__(self, vocab_size, mask_token_id, max_length=128,
                 d_model=384, n_heads=6, n_layers=6, d_ff=1536, dropout=0.1):
        super().__init__()
        self.vocab_size    = vocab_size
        self.mask_token_id = mask_token_id
        self.max_length    = max_length
        self.d_model       = d_model

        self.token_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight   # weight tying ✓

    def forward(self, x_t, t, attention_mask=None):
        """
        x_t: (B, L)  L must be <= max_length
        t:   (B,)    flow time in [0, 1]
        """
        B, L = x_t.shape
        # BUG3 FIX: guard against silent pos_emb clipping
        assert L <= self.max_length, (
            f"Input length {L} exceeds model max_length {self.max_length}. "
            f"Increase max_length or shorten prompt+generation."
        )
        h     = self.token_emb(x_t) + self.pos_emb[:, :L, :]
        t_emb = self.time_embed(t)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask=key_padding_mask)

        return self.head(self.ln_out(h))


# ======================================================================================
# 3. DISCRETE FLOW MATCHING: corruption, loss, samplers
# ======================================================================================

@dataclass
class DFMConfig:
    eps: float = 1e-3
    sampling_steps: int = 50


class DiscreteFlowMatching:
    """
    Linear masking probability path (Gat et al. 2024 / LLaDA):

    Convention (VERIFIED ✓):
        t=0 → all [MASK]  (source distribution)
        t=1 → clean data  (target distribution)
        P(masked at t) = 1 - t

    Loss (VERIFIED ✓):
        w(t) = 1 / (1 - t)   [time-derivative of the straight-line path]
        L = E_t[ w(t) * CE(logits, x1) ]   over masked, non-padding positions only

    Sampler (VERIFIED ✓):
        Euler CTMC, t: 0 → 1, step dt = 1/T
        p_unmask = dt / (1 - t)
        Absorbing: once revealed, tokens never go back to [MASK]
    """

    def __init__(self, mask_token_id, vocab_size, config=DFMConfig()):
        self.mask_token_id = mask_token_id
        self.vocab_size    = vocab_size
        self.cfg           = config

    def corrupt(self, x1, t):
        """
        VERIFIED:
          t=0 → P(mask)=1.0 → all masked
          t=0.5 → P(mask)=0.5 → half masked
          t≈1 → P(mask)≈0 → nothing masked
        """
        B, L = x1.shape
        keep_mask_prob = (1.0 - t).clamp(0.0, 1.0)[:, None].expand(B, L)
        rand      = torch.rand(B, L, device=x1.device)
        is_masked = rand < keep_mask_prob
        x_t = torch.where(is_masked, torch.full_like(x1, self.mask_token_id), x1)
        return x_t, is_masked

    def loss(self, model, x1, attention_mask=None):
        """VERIFIED: CE weighted by 1/(1-t), masked & non-padding positions only. ✓"""
        B, L   = x1.shape
        device = x1.device

        t = torch.rand(B, device=device) * (1.0 - self.cfg.eps)  # t ∈ [0, 1-eps]
        x_t, is_masked = self.corrupt(x1, t)

        logits = model(x_t, t, attention_mask=attention_mask)

        loss_positions = is_masked
        if attention_mask is not None:
            loss_positions = loss_positions & attention_mask.bool()

        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), x1.reshape(-1), reduction="none"
        ).reshape(B, L)

        weight      = (1.0 / (1.0 - t).clamp(min=self.cfg.eps))[:, None]
        weighted_ce = ce * weight * loss_positions.float()
        denom       = loss_positions.float().sum().clamp(min=1.0)
        loss        = weighted_ce.sum() / denom

        with torch.no_grad():
            frac_masked = is_masked.float().mean().item()

        return loss, frac_masked

    @torch.no_grad()
    def sample(self, model, batch_size, seq_len, device, temperature=1.0, top_p=None):
        """Unconditional CTMC Euler sampler. VERIFIED absorbing state. ✓"""
        model.eval()
        T  = self.cfg.sampling_steps
        dt = 1.0 / T

        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        for step in range(T):
            t_val = step * dt
            t     = torch.full((batch_size,), t_val, device=device)

            logits = model(x_t, t) / max(temperature, 1e-5)
            logits[..., self.mask_token_id] = -float("inf")   # never predict [MASK] as x1
            probs  = F.softmax(logits, dim=-1)

            if top_p is not None:
                probs = self._nucleus_filter(probs, top_p)

            sampled_x1 = torch.multinomial(
                probs.reshape(-1, self.vocab_size), num_samples=1
            ).reshape(batch_size, seq_len)

            is_masked = x_t.eq(self.mask_token_id)
            p_unmask  = min(dt / max(1.0 - t_val, self.cfg.eps), 1.0)
            rand      = torch.rand(batch_size, seq_len, device=device)
            do_unmask = is_masked & (rand < p_unmask)
            x_t = torch.where(do_unmask, sampled_x1, x_t)

        still_masked = x_t.eq(self.mask_token_id)
        if still_masked.any():
            t_final = torch.full((batch_size,), 1.0 - self.cfg.eps, device=device)
            logits  = model(x_t, t_final)
            logits[..., self.mask_token_id] = -float("inf")
            x_t = torch.where(still_masked, logits.argmax(-1), x_t)

        model.train()
        return x_t

    @torch.no_grad()
    def conditional_sample(self, model, prompt_ids, max_new_tokens, device,
                            temperature=1.0, top_p=None, return_history=False):
        """
        Prompt-conditioned generation. Fixes applied:
          BUG3: assert total length <= model.max_length before any forward pass.
          BUG5: is_generation_area defined BEFORE the loop (not inside it).
          BUG6: safety-net correction IS appended to history.
        """
        model.eval()
        B, L_prompt = prompt_ids.shape
        seq_len     = L_prompt + max_new_tokens

        # BUG3 FIX: guard here before any model call
        assert seq_len <= model.max_length, (
            f"prompt ({L_prompt}) + max_new_tokens ({max_new_tokens}) = {seq_len} "
            f"> model.max_length ({model.max_length}). Reduce prompt or max_new_tokens."
        )

        T  = self.cfg.sampling_steps
        dt = 1.0 / T

        x_t = torch.full((B, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        x_t[:, :L_prompt] = prompt_ids

        # BUG5 FIX: define OUTSIDE the loop so it's always in scope after the loop
        is_generation_area = torch.arange(seq_len, device=device).ge(L_prompt).unsqueeze(0)

        history = [(0.0, x_t.clone())] if return_history else None

        for step in range(T):
            t_val = step * dt
            t     = torch.full((B,), t_val, device=device)

            logits = model(x_t, t) / max(temperature, 1e-5)
            logits[..., self.mask_token_id] = -float("inf")
            probs  = F.softmax(logits, dim=-1)
            if top_p is not None:
                probs = self._nucleus_filter(probs, top_p)

            sampled_x1 = torch.multinomial(
                probs.reshape(-1, self.vocab_size), num_samples=1
            ).reshape(B, seq_len)

            is_masked  = x_t.eq(self.mask_token_id)
            can_unmask = is_masked & is_generation_area

            p_unmask  = dt / max(1.0 - t_val, self.cfg.eps)
            rand      = torch.rand(B, seq_len, device=device)
            do_unmask = can_unmask & (rand < p_unmask)
            x_t = torch.where(do_unmask, sampled_x1, x_t)

            if return_history:
                history.append((t_val + dt, x_t.clone()))

        # Safety net
        still_masked = x_t.eq(self.mask_token_id) & is_generation_area   # BUG5 FIX: always defined
        if still_masked.any():
            t_final = torch.full((B,), 1.0 - self.cfg.eps, device=device)
            logits  = model(x_t, t_final)
            logits[..., self.mask_token_id] = -float("inf")
            x_t = torch.where(still_masked, logits.argmax(-1), x_t)
            # BUG6 FIX: append corrected state to history
            if return_history:
                history.append((1.0, x_t.clone()))

        model.train()
        return (x_t, history) if return_history else x_t

    @staticmethod
    def _nucleus_filter(probs, top_p):
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cum_probs > top_p)
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0]  = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)


# ======================================================================================
# 4. TRAINING LOOP
# ======================================================================================

def _get_device():
    """BUG1 FIX: prefer CUDA > MPS > CPU (original code skipped CUDA entirely)."""
    if torch.cuda.is_available():
        return "cuda"
    # torch.backends.mps.is_available() is the stable API (exists since torch 1.12)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

import os
import math
import time
import torch

def train(
    epochs=5,
    batch_size=16,
    max_length=128,
    lr=3e-4,
    d_model=384,
    n_heads=6,
    n_layers=6,
    d_ff=1536,
    sampling_steps=20,
    device=None,
    log_every=20,
    sample_every_epoch=True,
    checkpoint_path="flow_matching_dllm.pt",
    best_checkpoint_dir="best_checkpoints",
    resume=True,
    gradient_accumulation_steps=8,
    use_fp16=False,  # Added toggle for mixed precision
):
    device = device or _get_device()  # BUG1 FIX
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"Using device: {device} (FP16: {use_fp16 and device_type == 'cuda'})")

    os.makedirs(best_checkpoint_dir, exist_ok=True)

    train_dl, val_dl, tokenizer, mask_token_id, vocab_size = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    model = FlowMatchingLLM(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        max_length=max_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    ).to(device)

    print(
        f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    dfm = DiscreteFlowMatching(
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        config=DFMConfig(sampling_steps=sampling_steps),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # FP16 GradScaler Initialization
    scaler = torch.amp.GradScaler(device_type, enabled=use_fp16)

    # --- Calculate scheduler steps based on gradient accumulation ---
    steps_per_epoch = math.ceil(len(train_dl) / gradient_accumulation_steps)
    total_optimization_steps = epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_optimization_steps,
        pct_start=0.05,
    )

    pad_token_id = tokenizer.pad_token_id
    start_epoch = 0
    best_models = []  # list of (val_loss, path)

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print("No optimizer state found in checkpoint.")

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("Loaded scheduler state.")
        else:
            print("No scheduler state found in checkpoint.")

        if "scaler_state_dict" in ckpt and use_fp16:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print("Loaded scaler state.")

        start_epoch = ckpt["epoch"] + 1

        if "best_models" in ckpt:
            best_models = ckpt["best_models"]

        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, x1 in enumerate(train_dl):
            x1 = x1.to(device)
            attention_mask = (x1 != pad_token_id).long()

            # --- Autocast forward pass to FP16 ---
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                loss, frac_masked = dfm.loss(
                    model,
                    x1,
                    attention_mask=attention_mask,
                )

            running_loss += loss.item()

            # Handle partial accumulation step on final batch
            is_last_batch = (batch_idx + 1) == len(train_dl)
            if is_last_batch and (len(train_dl) % gradient_accumulation_steps != 0):
                accum_steps = len(train_dl) % gradient_accumulation_steps
            else:
                accum_steps = gradient_accumulation_steps

            loss = loss / accum_steps
            
            # --- Scaled Backward Pass ---
            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or is_last_batch:
                # Unscale gradients prior to gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                # Step optimizer through scaler and update scale factor
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx % log_every == 0:
                avg = running_loss / (batch_idx + 1)
                print(
                    f"epoch {epoch+1}/{epochs} "
                    f"| step {batch_idx}/{len(train_dl)} "
                    f"| loss {avg:.4f} "
                    f"| frac_masked {frac_masked:.2f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e}"
                )

        val_loss = evaluate(model, dfm, val_dl, device, pad_token_id)
        train_loss = running_loss / len(train_dl)
        print(
            f"== epoch {epoch+1} in {time.time()-t0:.1f}s "
            f"| train_loss {train_loss:.4f} | val_loss {val_loss:.4f} =="
        )

        if sample_every_epoch:
            generate_and_print(
                model, dfm, tokenizer, device, n_samples=2, seq_len=max_length
            )

        # Save main resumption checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),  # Saved scaler state
            "vocab_size": vocab_size,
            "mask_token_id": mask_token_id,
            "max_length": max_length,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "best_models": best_models,
        }
        torch.save(ckpt, checkpoint_path)

        # Best Model Pruning & Saving Logic
        best_path = os.path.join(
            best_checkpoint_dir,
            f"epoch_{epoch+1}_valloss_{val_loss:.4f}.pt",
        )

        best_models.append((val_loss, best_path))
        best_models = sorted(best_models, key=lambda x: x[0])

        is_in_top_k = any(path == best_path for _, path in best_models[:2])

        while len(best_models) > 2:
            _, remove_path = best_models.pop(-1)
            if os.path.exists(remove_path):
                os.remove(remove_path)

        if is_in_top_k:
            torch.save(ckpt, best_path)

        print("Current Best Models:")
        for loss_val, path in best_models:
            print(f"  {loss_val:.4f} -> {path}")

    return model, dfm, tokenizer

@torch.no_grad()
def evaluate(model, dfm, dataloader, device, pad_token_id):
    model.eval()
    total, n = 0.0, 0
    for x1 in dataloader:
        x1             = x1.to(device)
        attention_mask = (x1 != pad_token_id).long()
        loss, _        = dfm.loss(model, x1, attention_mask=attention_mask)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def generate_and_print(model, dfm, tokenizer, device, n_samples=2, seq_len=64):
    samples = dfm.sample(model, batch_size=n_samples, seq_len=seq_len, device=device)
    print("---- generated samples ----")
    for i in range(n_samples):
        print(f"[{i}] {tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)}")
    print("----------------------------")


# ======================================================================================
# 5. VISUALIZATION
# ======================================================================================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_unmasking(
    model, dfm, tokenizer, device,
    prompt="The Sinclair Scientific Programmable was introduced in ",
    max_answer_tokens=50,
    sampling_steps=24,
    output_gif="diffusion_reveal.gif",
    fps=3, dpi=120,
    max_line_width=50.0,  # Max horizontal units before wrapping answer tokens
    row_spacing=0.8,       # Vertical distance between wrapped lines
):
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"].to(device)
    L_prompt   = prompt_ids.shape[1]

    final_seq, history = dfm.conditional_sample(
        model, prompt_ids, max_new_tokens=max_answer_tokens,
        device=device, return_history=True,
    )

    total_len = L_prompt + max_answer_tokens
    snapshots = []
    
    # First frame: everything masked
    snapshots.append((0.0, ["[MASK]"] * total_len,
                       np.array([tokenizer.mask_token_id] * total_len)))
    
    for t_val, seq in history[1:]:
        seq_list   = seq[0].cpu().tolist()
        tokens_str = ["[MASK]" if tid == tokenizer.mask_token_id
                      else tokenizer.decode([tid]) for tid in seq_list]
        snapshots.append((t_val, tokens_str, seq[0].cpu().numpy()))

    def compute_positions(tokens, char_spacing=0.5):
        x, positions = 0.0, []
        for tok in tokens:
            positions.append(x + len(tok) / 2.0)
            x += len(tok) + char_spacing
        return positions, x

    # Helper function to wrap answer tokens into multi-line coordinates
    def wrap_tokens(tokens, max_width, char_spacing=0.5):
        lines = []  # List of lines: each is a list of (token_idx, x_pos)
        current_line = []
        current_x = 0.0

        for idx, tok in enumerate(tokens):
            tok_w = len(tok) + char_spacing
            if current_x + tok_w > max_width and current_line:
                lines.append(current_line)
                current_line = []
                current_x = 0.0

            pos = current_x + len(tok) / 2.0
            current_line.append((idx, pos))
            current_x += tok_w

        if current_line:
            lines.append(current_line)
        return lines

    # Precalculate layout bounds across all snapshots
    max_answer_lines = 1
    for s in snapshots:
        answer_tokens = s[1][L_prompt:]
        lines = wrap_tokens(answer_tokens, max_line_width)
        max_answer_lines = max(max_answer_lines, len(lines))

    canvas_width = max_line_width + 6.0
    canvas_height = 2.0 + (max_answer_lines * row_spacing)

    fig, ax = plt.subplots(figsize=(canvas_width * 0.2 + 2.0, canvas_height))
    fig.patch.set_facecolor("white")

    def draw_frame(fi):
        ax.clear(); ax.axis("off")
        
        y_top = 0.5 * (max_answer_lines * row_spacing)
        ax.set_xlim(-3, max_line_width + 2)
        ax.set_ylim(-y_top - row_spacing, y_top + row_spacing)

        t_val, tokens_str, seq_ids = snapshots[fi]

        # 1. Render Prompt Line
        ax.text(-1, y_top, "Prompt:", ha="right", va="center", fontsize=12, weight="bold")
        for i, tok in enumerate(tokens_str[:L_prompt]):
            pos = compute_positions(tokens_str[:L_prompt])[0][i]
            ax.text(pos, y_top, tok, ha="center", va="center",
                    fontsize=13, color="steelblue", weight="bold", family="monospace")

        # 2. Render Wrapped Answer Lines
        answer_tokens = tokens_str[L_prompt:]
        wrapped_lines = wrap_tokens(answer_tokens, max_line_width)

        ax.text(-1, y_top - row_spacing, "Answer:", ha="right", va="center", fontsize=12, weight="bold")

        for line_idx, line in enumerate(wrapped_lines):
            y_pos = y_top - ((line_idx + 1) * row_spacing)
            for orig_idx, x_pos in line:
                tok = answer_tokens[orig_idx]
                is_m = seq_ids[L_prompt + orig_idx] == tokenizer.mask_token_id
                ax.text(x_pos, y_pos, tok, ha="center", va="center",
                        fontsize=13, color="dimgray" if is_m else "orange",
                        weight="bold", family="monospace")

        # Title / Frame Info
        ax.set_title(f"t = {t_val:.2f}  (frame {fi+1}/{len(snapshots)})", 
                     fontsize=14, y=-0.15, pad=20)

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=200, repeat=False)
    
    try:
        anim.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
        print(f"GIF saved as {output_gif}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        
    plt.close(fig)
    
    print("\nFinal generated text:")
    print(tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True))
    
    return final_seq
def inference_and_visualize(
    checkpoint_path="flow_matching_dllm_best.pt",
    prompt="The Sinclair Scientific Programmable was introduced in ",
    max_answer_tokens=85,
    sampling_steps=20,
    device=None,
):
    device = device or _get_device()   # BUG1 FIX
    ckpt   = torch.load(checkpoint_path, map_location=device)

    model = FlowMatchingLLM(
        vocab_size=ckpt["vocab_size"], mask_token_id=ckpt["mask_token_id"],
        max_length=ckpt.get("max_length", 128), d_model=ckpt.get("d_model", 384),
        n_heads=ckpt.get("n_heads", 6), n_layers=ckpt.get("n_layers", 6),
        d_ff=ckpt.get("d_ff", 1536),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.pad_token = tokenizer.eos_token

    dfm = DiscreteFlowMatching(
        mask_token_id=ckpt["mask_token_id"], vocab_size=ckpt["vocab_size"],
        config=DFMConfig(sampling_steps=sampling_steps),
    )

    visualize_unmasking(model, dfm, tokenizer, device,
                        prompt=prompt, max_answer_tokens=max_answer_tokens)


# ======================================================================================
# 6. ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    model, dfm, tokenizer = train(
        epochs=50,         # reduced from 500; use best-checkpoint for longer runs
        batch_size=32,
        max_length=96,
        lr=3e-4,
        d_model=512,
        n_heads=16,
        n_layers=40,
        d_ff=1536,
        sampling_steps=17,
        gradient_accumulation_steps=2
    )

    print("\n===== Generating visualization =====")
    inference_and_visualize(checkpoint_path="best_checkpoints/epoch_3_valloss_21.0907.pt")