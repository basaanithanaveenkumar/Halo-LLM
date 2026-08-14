"""
Flow Matching Discrete Diffusion Language Model  ── FINAL (verified + overfit flag)
====================================================================================
All previously found bugs are fixed. An OVERFIT_TEXT flag has been added so you
can verify the model can actually learn before committing to full pre-training.

NEW: overfit_text parameter in train()
--------------------------------------
Set  overfit_text="my name is naveen I am an ML engineer"  (or any string) to run
in fast-sanity / memorisation mode:

  • The WikiText-2 dataloader is BYPASSED entirely.
  • A tiny in-memory dataset is built by repeating the target sentence to fill
    `n_overfit_copies` sequences, each padded/truncated to max_length.
  • batch_size is forced to min(batch_size, n_overfit_copies) so you always get
    a full batch.
  • At the end of every epoch a RECONSTRUCT check is run: the sentence is fully
    masked (t=0) and the model is asked to reconstruct it. The exact-match
    accuracy is printed so you can watch memorisation happen in real time.
  • Good hyperparams for overfit: lr=1e-3, epochs=300, d_model=128, n_layers=4.
    You should see reconstruct accuracy hit 100% within ~50 epochs on a single
    sentence.

Set  overfit_text=None  (default) to run normal pre-training on WikiText-2.
"""

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ======================================================================================
# 1. DATALOADER
# ======================================================================================

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer_name="gpt2", max_length=128, split="train", size=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        self.max_length = max_length
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        if size is not None:
            dataset = dataset.select(range(min(size, len(dataset))))

        self.data          = dataset
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size    = len(self.tokenizer)

        print(f"Loaded {len(self.data)} samples from WikiText-2 ({split} split).")
        print(f"Vocabulary size (incl. [MASK]): {self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        if not text or text.strip() == "":
            text = " "
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
            add_special_tokens=False,
        )
        return encoding["input_ids"].squeeze(0)


def get_dataloaders(batch_size=16, max_length=128, train_size=2048, val_size=512,
                    tokenizer_name="gpt2"):
    train_dataset = WikiTextDataset(tokenizer_name, max_length, "train",      train_size)
    val_dataset   = WikiTextDataset(tokenizer_name, max_length, "validation", val_size)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)
    return (train_dl, val_dl,
            train_dataset.tokenizer, train_dataset.mask_token_id, train_dataset.vocab_size)


# ======================================================================================
# 1b. OVERFIT DATALOADER  (bypasses WikiText entirely)
# ======================================================================================

def build_overfit_dataloader(text, tokenizer, max_length, batch_size,
                              n_overfit_copies=256):
    """
    Build a tiny dataloader that contains only `n_overfit_copies` copies of a
    single tokenised sentence. Drop-last is False so every copy is seen.

    Args:
        text:             The sentence to memorise, e.g. "my name is naveen".
        tokenizer:        A fully-set-up tokenizer (mask_token already added).
        max_length:       Sequence length the model was built for.
        batch_size:       Desired batch size (clamped to n_overfit_copies).
        n_overfit_copies: How many times to repeat the sentence in the dataset.
    Returns:
        DataLoader, token_ids_1d (LongTensor of shape (max_length,))
    """
    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
        add_special_tokens=False,
    )
    ids_1d = enc["input_ids"].squeeze(0)          # (max_length,)
    ids_nd = ids_1d.unsqueeze(0).repeat(n_overfit_copies, 1)  # (N, max_length)

    dataset = TensorDataset(ids_nd)
    bs      = min(batch_size, n_overfit_copies)
    dl      = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    print(f"[OVERFIT MODE] text: '{text}'")
    print(f"[OVERFIT MODE] token ids: {ids_1d.tolist()}")
    print(f"[OVERFIT MODE] dataset: {n_overfit_copies} copies, batch_size={bs}")
    return dl, ids_1d


# ======================================================================================
# 2. MODEL
# ======================================================================================

class SinusoidalTimeEmbedding(nn.Module):
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
    """Pre-norm bidirectional block, AdaLN-Zero time conditioning."""
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
        return x + self.ff(h)


class FlowMatchingLLM(nn.Module):
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
        self.head.weight = self.token_emb.weight   # weight tying

    def forward(self, x_t, t, attention_mask=None):
        B, L = x_t.shape
        assert L <= self.max_length, (
            f"Input length {L} > model max_length {self.max_length}."
        )
        h     = self.token_emb(x_t) + self.pos_emb[:, :L, :]
        t_emb = self.time_embed(t)
        kpm   = (attention_mask == 0) if attention_mask is not None else None
        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask=kpm)
        return self.head(self.ln_out(h))


# ======================================================================================
# 3. DISCRETE FLOW MATCHING
# ======================================================================================

@dataclass
class DFMConfig:
    eps: float = 1e-3
    sampling_steps: int = 50


class DiscreteFlowMatching:
    """
    Linear masking path: t=0 → all [MASK], t=1 → clean data.
    P(masked | t) = 1 - t.
    Loss weight: 1 / (1 - t).
    Sampler: CTMC Euler, p_unmask = dt / (1 - t).
    """

    def __init__(self, mask_token_id, vocab_size, config=DFMConfig()):
        self.mask_token_id = mask_token_id
        self.vocab_size    = vocab_size
        self.cfg           = config

    def corrupt(self, x1, t):
        B, L = x1.shape
        keep_mask_prob = (1.0 - t).clamp(0.0, 1.0)[:, None].expand(B, L)
        is_masked = torch.rand(B, L, device=x1.device) < keep_mask_prob
        x_t = torch.where(is_masked, torch.full_like(x1, self.mask_token_id), x1)
        return x_t, is_masked

    def loss(self, model, x1, attention_mask=None):
        B, L   = x1.shape
        device = x1.device
        t      = torch.rand(B, device=device) * (1.0 - self.cfg.eps)
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
        loss        = weighted_ce.sum() / loss_positions.float().sum().clamp(min=1.0)

        return loss, is_masked.float().mean().item()

    @torch.no_grad()
    def reconstruct_from_mask(self, model, x1_ids, device, temperature=1.0):
        """
        OVERFIT CHECK: fully mask the input, ask the model to reconstruct it in
        one greedy forward pass (t ≈ 0, so the model has seen nothing revealed).
        Returns predicted token ids and per-token exact-match accuracy (over
        non-padding positions).
        """
        model.eval()
        B, L = x1_ids.shape
        x_masked = torch.full_like(x1_ids, self.mask_token_id, device=device)
        t_near0  = torch.full((B,), self.cfg.eps, device=device)

        logits = model(x_masked.to(device), t_near0)
        logits[..., self.mask_token_id] = -float("inf")
        preds = logits.argmax(dim=-1)           # (B, L)

        # accuracy only on non-pad positions
        non_pad = (x1_ids.to(device) != model.mask_token_id)
        correct = (preds == x1_ids.to(device)) & non_pad
        acc     = correct.float().sum() / non_pad.float().sum().clamp(min=1.0)
        model.train()
        return preds, acc.item()

    @torch.no_grad()
    def sample(self, model, batch_size, seq_len, device, temperature=1.0, top_p=None):
        model.eval()
        T  = self.cfg.sampling_steps
        dt = 1.0 / T
        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        for step in range(T):
            t_val = step * dt
            t     = torch.full((batch_size,), t_val, device=device)
            logits = model(x_t, t) / max(temperature, 1e-5)
            logits[..., self.mask_token_id] = -float("inf")
            probs  = F.softmax(logits, dim=-1)
            if top_p is not None:
                probs = self._nucleus_filter(probs, top_p)
            sampled_x1 = torch.multinomial(
                probs.reshape(-1, self.vocab_size), 1
            ).reshape(batch_size, seq_len)
            is_masked = x_t.eq(self.mask_token_id)
            p_unmask  = min(dt / max(1.0 - t_val, self.cfg.eps), 1.0)
            do_unmask = is_masked & (torch.rand_like(is_masked.float()) < p_unmask)
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
        model.eval()
        B, L_prompt = prompt_ids.shape
        seq_len     = L_prompt + max_new_tokens
        assert seq_len <= model.max_length, (
            f"prompt ({L_prompt}) + max_new_tokens ({max_new_tokens}) = {seq_len} "
            f"> model.max_length ({model.max_length})."
        )
        T  = self.cfg.sampling_steps
        dt = 1.0 / T
        x_t = torch.full((B, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        x_t[:, :L_prompt] = prompt_ids

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
                probs.reshape(-1, self.vocab_size), 1
            ).reshape(B, seq_len)
            can_unmask = x_t.eq(self.mask_token_id) & is_generation_area
            p_unmask   = dt / max(1.0 - t_val, self.cfg.eps)
            do_unmask  = can_unmask & (torch.rand(B, seq_len, device=device) < p_unmask)
            x_t = torch.where(do_unmask, sampled_x1, x_t)
            if return_history:
                history.append((t_val + dt, x_t.clone()))

        still_masked = x_t.eq(self.mask_token_id) & is_generation_area
        if still_masked.any():
            t_final = torch.full((B,), 1.0 - self.cfg.eps, device=device)
            logits  = model(x_t, t_final)
            logits[..., self.mask_token_id] = -float("inf")
            x_t = torch.where(still_masked, logits.argmax(-1), x_t)
            if return_history:
                history.append((1.0, x_t.clone()))

        model.train()
        return (x_t, history) if return_history else x_t

    @staticmethod
    def _nucleus_filter(probs, top_p):
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff    = (cum_probs > top_p)
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0]  = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)


# ======================================================================================
# 4. TRAINING LOOP  (overfit_text flag lives here)
# ======================================================================================

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    # ── Overfit flag ──────────────────────────────────────────────────────────
    overfit_text=None,
    # ^ Set to a string like "my name is naveen I am an ML engineer" to enable
    #   memorisation mode. Set to None for full WikiText-2 pre-training.
    n_overfit_copies=256,
    # ^ How many times to repeat the sentence in the tiny dataset.
    #   256 copies with batch_size=16 -> 16 gradient steps per epoch.

    # ── General hyperparams ───────────────────────────────────────────────────
    epochs=300,
    batch_size=16,
    max_length=128,
    lr=1e-3,                 # higher LR for overfit; lower (3e-4) for pretraining
    d_model=128,             # small for overfit; 384+ for pretraining
    n_heads=4,               # must divide d_model
    n_layers=4,              # shallow for overfit; 6+ for pretraining
    d_ff=512,
    dropout=0.0,             # 0.0 for overfit (no regularisation); 0.1 for pretraining
    sampling_steps=50,
    device=None,
    log_every=50,
    sample_every_epoch=True,
    checkpoint_path="flow_matching_dllm.pt",
    best_checkpoint_path="flow_matching_dllm_best.pt",

    # ── WikiText pre-training args (ignored in overfit mode) ──────────────────
    train_size=2048,
    val_size=512,
):
    device = device or _get_device()
    print(f"Using device: {device}")

    # ── 1. Build tokenizer once (shared across all dataloaders) ───────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    mask_token_id = tokenizer.mask_token_id
    vocab_size    = len(tokenizer)
    pad_token_id  = tokenizer.pad_token_id

    # ── 2. Build dataloader(s) ────────────────────────────────────────────────
    if overfit_text is not None:
        # ── OVERFIT MODE ──
        train_dl, overfit_ids = build_overfit_dataloader(
            overfit_text, tokenizer, max_length, batch_size, n_overfit_copies
        )
        val_dl      = None  # no validation in overfit mode
        overfit_ids = overfit_ids.unsqueeze(0).to(device)  # (1, max_length) for reconstruct check
    else:
        # ── PRETRAINING MODE ──
        from datasets import load_dataset as _ld
        train_dl, val_dl, tokenizer, mask_token_id, vocab_size = get_dataloaders(
            batch_size=batch_size, max_length=max_length,
            train_size=train_size, val_size=val_size,
        )
        pad_token_id  = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id
        vocab_size    = len(tokenizer)
        overfit_ids   = None

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = FlowMatchingLLM(
        vocab_size=vocab_size, mask_token_id=mask_token_id, max_length=max_length,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M parameters")

    dfm = DiscreteFlowMatching(
        mask_token_id=mask_token_id, vocab_size=vocab_size,
        config=DFMConfig(sampling_steps=sampling_steps),
    )

    # ── 4. Optimiser + schedule ───────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0 if overfit_text else 0.01)
    total_steps = epochs * len(train_dl)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05
    )
    best_val = float("inf")

    # ── 5. Training epochs ────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        t0           = time.time()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_dl):
            # TensorDataset wraps items in a list; WikiTextDataset returns a tensor directly
            x1 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x1 = x1.to(device)
            attention_mask = (x1 != pad_token_id).long()

            loss, frac_masked = dfm.loss(model, x1, attention_mask=attention_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # ── per-epoch logging ──────────────────────────────────────────────────
        avg_loss = running_loss / len(train_dl)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t0

            if overfit_text is not None:
                # ── OVERFIT: reconstruct check ──
                preds, acc = dfm.reconstruct_from_mask(model, overfit_ids, device)
                pred_text  = tokenizer.decode(preds[0].tolist(), skip_special_tokens=True)
                print(f"epoch {epoch+1:4d}/{epochs} | loss {avg_loss:.4f} "
                      f"| reconstruct_acc {acc*100:.1f}% | {elapsed:.1f}s")
                print(f"  target : '{overfit_text}'")
                print(f"  predict: '{pred_text.strip()}'")
                if acc >= 1.0:
                    print("  ✓ PERFECT MEMORISATION — stopping early.")
                    break
            else:
                # ── PRETRAINING: val loss ──
                val_loss = evaluate(model, dfm, val_dl, device, pad_token_id)
                print(f"epoch {epoch+1:4d}/{epochs} | train {avg_loss:.4f} "
                      f"| val {val_loss:.4f} | {elapsed:.1f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        dict(model_state_dict=model.state_dict(), epoch=epoch,
                             vocab_size=vocab_size, mask_token_id=mask_token_id,
                             max_length=max_length, d_model=d_model, n_heads=n_heads,
                             n_layers=n_layers, d_ff=d_ff),
                        best_checkpoint_path,
                    )
                    print(f"  ↑ best val={best_val:.4f} → saved {best_checkpoint_path}")

                if sample_every_epoch:
                    generate_and_print(model, dfm, tokenizer, device,
                                       n_samples=2, seq_len=max_length)

        # ── checkpoint every epoch (overfit + pretraining) ────────────────────
        ckpt = dict(model_state_dict=model.state_dict(), epoch=epoch,
                    vocab_size=vocab_size, mask_token_id=mask_token_id,
                    max_length=max_length, d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers, d_ff=d_ff)
        torch.save(ckpt, checkpoint_path)

    return model, dfm, tokenizer


# ======================================================================================
# 5. HELPERS
# ======================================================================================

@torch.no_grad()
def evaluate(model, dfm, dataloader, device, pad_token_id):
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        x1 = batch[0] if isinstance(batch, (list, tuple)) else batch
        x1 = x1.to(device)
        attention_mask = (x1 != pad_token_id).long()
        loss, _ = dfm.loss(model, x1, attention_mask=attention_mask)
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
# 6. VISUALIZATION
# ======================================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

def visualize_unmasking(
    model,
    dfm,
    tokenizer,
    device,
    prompt="Hi who are you?  ",
    max_answer_tokens=50,
    sampling_steps=80,
    output_gif="diffusion_reveal_grid.gif",
    fps=3,
    dpi=120,
    cols=10,                 # max words per row
    cell_size=0.8,           # size of each square cell in inches
):
    # 1. Tokenize prompt and run conditional sampling with history
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"].to(device)
    L_prompt = prompt_ids.shape[1]

    final_seq, history = dfm.conditional_sample(
        model,
        prompt_ids,
        max_new_tokens=max_answer_tokens,
        device=device,
        return_history=True,
    )

    total_len = L_prompt + max_answer_tokens

    # 2. Build snapshots: (t, list_of_token_strings, array_of_token_ids)
    snapshots = []
    for t_val, seq in history:   # history includes t=0 snapshot
        seq_list = seq[0].cpu().tolist()
        tokens_str = [
            "[MASK]" if tid == tokenizer.mask_token_id
            else tokenizer.decode([tid])
            for tid in seq_list
        ]
        snapshots.append((t_val, tokens_str, seq[0].cpu().numpy()))

    # 3. Determine grid dimensions
    rows = int(np.ceil(total_len / cols))
    # Pad the sequence to fill the grid (for alignment)
    pad_len = rows * cols - total_len
    padded_tokens = [""] * pad_len   # empty strings for padding cells

    # 4. Set up the figure
    fig_width = cols * cell_size + 2.0   # extra for padding and title
    fig_height = rows * cell_size + 2.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. Drawing function
    def draw_frame(fi):
        ax.clear()
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        t_val, tokens_str, seq_ids = snapshots[fi]
        # Pad the token list to fill the grid
        full_tokens = tokens_str + padded_tokens
        full_ids = list(seq_ids) + [tokenizer.pad_token_id] * pad_len

        # Draw each cell
        for idx, (tok, tid) in enumerate(zip(full_tokens, full_ids)):
            if tok == "":
                continue   # skip padding
            row = idx // cols
            col = idx % cols
            x = col
            y = rows - 1 - row   # so row 0 is at top

            is_prompt = idx < L_prompt
            is_mask = (tid == tokenizer.mask_token_id)

            # Determine cell background and text color
            if is_prompt:
                facecolor = "none"
                edgecolor = "none"
                text_color = "darkblue"
                weight = "bold"
            else:
                if is_mask:
                    facecolor = "#D6EAF8"   # light muted blue
                    edgecolor = "#A9CCE3"
                    text_color = "gray"
                    weight = "normal"
                else:
                    facecolor = "none"
                    edgecolor = "none"
                    text_color = "orange"
                    weight = "bold"

            # Draw background rectangle if it's a mask
            if is_mask and not is_prompt:
                rect = Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    facecolor=facecolor, edgecolor=edgecolor, linewidth=1
                )
                ax.add_patch(rect)

            # Draw token text
            ax.text(
                x, y, tok,
                ha="center", va="center",
                fontsize=14 if not is_prompt else 12,   # prompt slightly smaller to fit
                color=text_color,
                weight=weight,
                family="sans-serif"
            )

        # Title with current time
        ax.set_title(
            f"Discrete Flow Matching – t = {t_val:.2f}  (frame {fi+1}/{len(snapshots)})",
            fontsize=16, pad=20
        )

    # 6. Create and save animation
    anim = animation.FuncAnimation(
        fig, draw_frame,
        frames=len(snapshots),
        interval=200,   # ms per frame
        repeat=True     # loop forever
    )

    try:
        anim.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
        print(f"GIF saved as {output_gif}")
    except Exception as e:
        print(f"Could not save GIF: {e}")

    plt.close(fig)

    # 7. Print final text
    final_text = tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True)
    print("\nFinal generated text:")
    print(final_text)
    return final_seq

def inference_and_visualize(
    checkpoint_path="flow_matching_dllm_best.pt",
    prompt="Hi who are you?",
    max_answer_tokens=18,
    sampling_steps=50,
    device=None,
):
    device = device or _get_device()
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model  = FlowMatchingLLM(
        vocab_size=ckpt["vocab_size"], mask_token_id=ckpt["mask_token_id"],
        max_length=ckpt.get("max_length", 128), d_model=ckpt.get("d_model", 384),
        n_heads=ckpt.get("n_heads", 6),         n_layers=ckpt.get("n_layers", 6),
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
# 7. ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true",
                        help="Run in overfit/sanity mode on a single sentence")
    parser.add_argument("--text", type=str,
                        default=" Hi who are you? I am Halo, a flow matching based  llm model",
                        help="Sentence to memorise in overfit mode")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr",     type=float, default=None)
    args = parser.parse_args()

    if args.overfit:
        # ── OVERFIT MODE ────────────────────────────────────────────────────────
        # Fast sanity check: a tiny model should memorise one sentence in <1 min.
        # Expected output: reconstruct_acc hits 100% within ~50-150 epochs.
        # print("=" * 60)
        # print("OVERFIT / SANITY MODE")
        # print(f"  target: '{args.text}'")
        # print("=" * 60)
        # model, dfm, tokenizer = train(
        #     overfit_text=args.text,
        #     n_overfit_copies=256,
        #     epochs=args.epochs or 80,
        #     batch_size=1,
        #     max_length=64,          # short: the sentence fits in <<64 tokens
        #     lr=args.lr or 1e-4,
        #     d_model=512,
        #     n_heads=8,
        #     n_layers=8,
        #     d_ff=512,
        #     dropout=0.2,            # no dropout: we WANT to overfit
        #     sampling_steps=40,
        #     log_every=10,           # print every 10 epochs
            
        # )
        print("\n===== Generating visualization =====")
        inference_and_visualize(checkpoint_path="flow_matching_dllm.pt")
    else:
        # ── PRE-TRAINING MODE ────────────────────────────────────────────────────
        print("=" * 60)
        print("PRE-TRAINING MODE  (WikiText-2)")
        print("=" * 60)
        model, dfm, tokenizer = train(
            overfit_text=None,
            epochs=args.epochs or 20,
            batch_size=16,
            max_length=128,
            lr=args.lr or 3e-4,
            d_model=384,
            n_heads=6,
            n_layers=6,
            d_ff=1536,
            dropout=0.1,
            sampling_steps=50,
            train_size=2048,
            val_size=512,
            log_every=1,
            sample_every_epoch=True,
        )
        print("\n===== Generating visualization =====")
        inference_and_visualize(checkpoint_path="flow_matching_dllm.pt")