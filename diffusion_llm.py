"""
Pure Masked Discrete Diffusion Language Model (MDLM / D3PM absorbing-state)
============================================================================
This is the "pure diffusion" sibling of the flow-matching and block-diffusion
files: ONE global time t per sequence, bidirectional attention over the WHOLE
sequence at once (no blocks, no autoregressive-across-blocks structure), and
a proper continuous-time diffusion derivation (forward process, NELBO, and an
analytic ancestral reverse sampler) rather than the flow-matching / CTMC
framing used in the other file.

Lineage: Austin et al. 2021 "D3PM" (absorbing-state discrete diffusion) ->
Sahoo et al. 2024 "MDLM" (Simplified, effective continuous-time masked
diffusion LM) -> Nie et al. 2024 "LLaDA" (scaled up to a full LLM).

--------------------------------------------------------------------------
FORWARD PROCESS
--------------------------------------------------------------------------
t in [0, 1], t=0 -> clean data, t=1 -> fully [MASK]'d (absorbing state).
A noise schedule alpha(t) gives the probability a token is STILL the
original (unmasked) token at time t:
    alpha(0) = 1   (no corruption)
    alpha(1) = 0   (fully corrupted)
    alpha(t) monotonically decreasing

    q(x_t = [MASK] | x_0)  = 1 - alpha(t)
    q(x_t = x_0     | x_0) = alpha(t)

Once a token is masked it can never become unmasked again under the forward
process (that's what "absorbing state" means) -- exactly mirrors an
irrecoverable corruption process, the discrete analogue of the noise you'd
add in a continuous diffusion model.

--------------------------------------------------------------------------
TRAINING OBJECTIVE (continuous-time NELBO, "SUBS" parameterization)
--------------------------------------------------------------------------
Sahoo et al. show the (Rao-Blackwellized, variance-reduced) continuous-time
ELBO for this forward process collapses to a beautifully simple form: a
cross-entropy loss on masked positions only, weighted by the schedule's
instantaneous corruption rate:

    L = E_t E_{x_t~q(.|x_0)} [ w(t) * (-log p_theta(x_0 | x_t, t)) ]   , only at masked positions

    w(t) = -alpha'(t) / (1 - alpha(t))

This is model-schedule-agnostic: any monotonically-decreasing alpha(t) with
alpha(0)=1, alpha(1)=0 gives a valid NELBO via this same formula, which is
why this file implements it via a pluggable `NoiseSchedule` class rather
than hard-coding one schedule (log-linear "linear-masking" schedule and
cosine schedule are both provided below).

--------------------------------------------------------------------------
REVERSE PROCESS / SAMPLING (analytic posterior, absorbing-state)
--------------------------------------------------------------------------
Because the forward process is absorbing (mask is a sink state), the true
reverse posterior q(x_s | x_t, x_0) for s < t has a closed form (no need to
learn it directly) -- only x_0 needs to be estimated by the network:

    If x_t is already unmasked: x_s = x_t deterministically (nothing to do).
    If x_t is [MASK]:
        P(x_s = [MASK] | x_t = [MASK]) = (1 - alpha_s) / (1 - alpha_t)
        P(x_s = x_0    | x_t = [MASK]) = (alpha_s - alpha_t) / (1 - alpha_t)
                                          , x_0 drawn from p_theta(x_0 | x_t, t)

Sampling simulates this from t=1 (all [MASK]) down to t=0 (clean) over T
discrete steps, substituting the network's predicted p_theta(x_0|x_t,t) for
the true x_0 at each step -- standard ancestral sampling for discrete
diffusion.
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


# ======================================================================================
# 1. DATALOADER (self-contained WikiText-2 loader, same pattern as the other files)
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
        self.data = dataset
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)

        print(f"Loaded {len(self.data)} samples from WikiText-2 ({split} split).")
        print(f"Vocabulary size (incl. [MASK]): {self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        if not text or text.strip() == "":
            text = "[PAD]"
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0)


def get_dataloaders(batch_size=16, max_length=128, train_size=2048, val_size=512,
                     tokenizer_name="gpt2"):
    train_dataset = WikiTextDataset(tokenizer_name, max_length, "train", train_size)
    val_dataset = WikiTextDataset(tokenizer_name, max_length, "validation", val_size)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dl, val_dl, train_dataset.tokenizer, train_dataset.mask_token_id, train_dataset.vocab_size


# ======================================================================================
# 2. NOISE SCHEDULES: alpha(t) and its derivative, pluggable
# ======================================================================================

class NoiseSchedule(ABC):
    """alpha(t): probability a token is STILL clean (unmasked) at time t.
    Must satisfy alpha(0)=1, alpha(1)=0, monotonically decreasing."""

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor: ...


class LogLinearSchedule(NoiseSchedule):
    """The schedule used in the original MDLM paper: alpha(t) = 1 - t.
    Masking probability grows linearly with t; loss weight w(t) = 1/t."""

    def alpha(self, t):
        return 1.0 - t

    def alpha_prime(self, t):
        return -torch.ones_like(t)


class CosineSchedule(NoiseSchedule):
    """A smoother schedule (spends more time near both endpoints, less in the
    middle) commonly used in continuous diffusion models: alpha(t) = cos^2(pi*t/2)."""

    def alpha(self, t):
        return torch.cos(0.5 * math.pi * t) ** 2

    def alpha_prime(self, t):
        # d/dt cos^2(pi t / 2) = -pi/2 * sin(pi t)
        return -0.5 * math.pi * torch.sin(math.pi * t)


# ======================================================================================
# 3. MODEL: bidirectional transformer predicting p_theta(x_0 | x_t, t)
# ======================================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional (fully non-causal) transformer block, AdaLN-Zero
    conditioned on a single global time t per sequence."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
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


class MaskedDiffusionLM(nn.Module):
    """Bidirectional denoiser: predicts p_theta(x_0 | x_t, t) for every position,
    conditioning on the WHOLE (partially masked) sequence at once -- no blocks,
    no causal structure of any kind. This is the "pure diffusion" architecture."""
    def __init__(self, vocab_size, mask_token_id, max_length=128,
                 d_model=384, n_heads=6, n_layers=6, d_ff=1536, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(self, x_t, t, attention_mask=None):
        """
        x_t: (B, L) partially-masked token ids
        t:   (B,)   global diffusion time in [0, 1] for the WHOLE sequence
        attention_mask: (B, L) 1 = real token, 0 = padding
        returns logits: (B, L, V)
        """
        B, L = x_t.shape
        h = self.token_emb(x_t) + self.pos_emb[:, :L, :]
        t_emb = self.time_embed(t)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask=key_padding_mask)

        h = self.ln_out(h)
        return self.head(h)


# ======================================================================================
# 4. FORWARD PROCESS / NELBO LOSS / ANCESTRAL SAMPLER
# ======================================================================================

@dataclass
class MDLMConfig:
    eps: float = 1e-3          # sample t in [eps, 1] to avoid the w(t) blow-up at t=0
    sampling_steps: int = 50   # number of ancestral reverse-diffusion steps


class MaskedDiffusion:
    """Wraps a NoiseSchedule with the forward corruption process, the NELBO loss,
    and the analytic ancestral reverse sampler for absorbing-state discrete diffusion."""

    def __init__(self, mask_token_id, vocab_size, schedule: NoiseSchedule, config=MDLMConfig()):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.schedule = schedule
        self.cfg = config

    # ---- forward process: q(x_t | x_0) ------------------------------------------------
    def corrupt(self, x0, t):
        """
        x0: (B, L) clean token ids
        t:  (B,)   diffusion time in [0, 1]
        returns x_t: (B, L), is_masked: (B, L) bool
        """
        B, L = x0.shape
        alpha_t = self.schedule.alpha(t).clamp(0.0, 1.0)          # (B,) prob. STILL clean
        mask_prob = (1.0 - alpha_t)[:, None].expand(B, L)
        rand = torch.rand(B, L, device=x0.device)
        is_masked = rand < mask_prob
        x_t = torch.where(is_masked, torch.full_like(x0, self.mask_token_id), x0)
        return x_t, is_masked

    # ---- NELBO loss: weighted cross entropy on masked positions -----------------------
    def loss(self, model, x0, attention_mask=None):
        B, L = x0.shape
        device = x0.device

        t = torch.rand(B, device=device) * (1.0 - self.cfg.eps) + self.cfg.eps  # t in [eps, 1]
        x_t, is_masked = self.corrupt(x0, t)

        logits = model(x_t, t, attention_mask=attention_mask)  # (B, L, V)

        loss_positions = is_masked
        if attention_mask is not None:
            loss_positions = loss_positions & attention_mask.bool()

        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), x0.reshape(-1), reduction="none"
        ).reshape(B, L)

        alpha_t = self.schedule.alpha(t).clamp(0.0, 1.0)
        alpha_prime_t = self.schedule.alpha_prime(t)
        weight = (-alpha_prime_t / (1.0 - alpha_t).clamp(min=self.cfg.eps))[:, None]  # (B, 1)

        weighted_ce = ce * weight * loss_positions.float()
        denom = loss_positions.float().sum().clamp(min=1.0)
        loss = weighted_ce.sum() / denom

        with torch.no_grad():
            frac_masked = is_masked.float().mean().item()

        return loss, frac_masked

    # ---- reverse process: analytic ancestral sampler -----------------------------------
    @torch.no_grad()
    def sample(self, model, batch_size, seq_len, device, temperature=1.0):
        """
        Simulates the reverse chain from t=1 (all [MASK]) down to t=0 (clean) over
        `sampling_steps` discrete steps, using the CLOSED-FORM absorbing-state
        posterior with the network's predicted x_0 substituted in (see module
        docstring). This is the standard MDLM/D3PM ancestral sampler.
        """
        model.eval()
        T = self.cfg.sampling_steps
        t_steps = torch.linspace(1.0, self.cfg.eps, T + 1, device=device)  # t: 1 -> ~0

        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        for step in range(T):
            t_cur = t_steps[step]
            t_next = t_steps[step + 1]
            t_batch = t_cur.expand(batch_size)

            logits = model(x_t, t_batch, attention_mask=None) / max(temperature, 1e-5)
            logits[..., self.mask_token_id] = -float("inf")  # never predict [MASK] as x_0
            probs = F.softmax(logits, dim=-1)
            x0_pred = torch.multinomial(
                probs.reshape(-1, self.vocab_size), num_samples=1
            ).reshape(batch_size, seq_len)

            alpha_t = self.schedule.alpha(t_cur).clamp(0.0, 1.0)
            alpha_s = self.schedule.alpha(t_next).clamp(0.0, 1.0)

            # P(reveal now | still masked) = (alpha_s - alpha_t) / (1 - alpha_t)
            denom = (1.0 - alpha_t).clamp(min=1e-8)
            p_reveal = ((alpha_s - alpha_t) / denom).clamp(0.0, 1.0)

            is_masked = x_t.eq(self.mask_token_id)
            rand = torch.rand(batch_size, seq_len, device=device)
            do_reveal = is_masked & (rand < p_reveal)

            x_t = torch.where(do_reveal, x0_pred, x_t)

        # safety net: force-fill any residual [MASK] at t~eps
        still_masked = x_t.eq(self.mask_token_id)
        if still_masked.any():
            t_final = torch.full((batch_size,), self.cfg.eps, device=device)
            logits = model(x_t, t_final)
            logits[..., self.mask_token_id] = -float("inf")
            final_pred = logits.argmax(dim=-1)
            x_t = torch.where(still_masked, final_pred, x_t)

        model.train()
        return x_t


# ======================================================================================
# 5. TRAINING LOOP
# ======================================================================================

def train(
    epochs=5, batch_size=16, max_length=128, lr=3e-4,
    d_model=384, n_heads=6, n_layers=6, d_ff=1536,
    sampling_steps=50, schedule_name="log_linear",
    device=None, log_every=20, sample_every_epoch=True,
    checkpoint_path="pure_diffusion_llm.pt",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, val_dl, tokenizer, mask_token_id, vocab_size = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    model = MaskedDiffusionLM(
        vocab_size=vocab_size, mask_token_id=mask_token_id, max_length=max_length,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters | schedule={schedule_name}")

    schedule = {"log_linear": LogLinearSchedule, "cosine": CosineSchedule}[schedule_name]()
    diffusion = MaskedDiffusion(
        mask_token_id=mask_token_id, vocab_size=vocab_size, schedule=schedule,
        config=MDLMConfig(sampling_steps=sampling_steps),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05)

    pad_token_id = tokenizer.pad_token_id
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch_idx, x0 in enumerate(train_dl):
            x0 = x0.to(device)
            attention_mask = (x0 != pad_token_id).long()

            loss, frac_masked = diffusion.loss(model, x0, attention_mask=attention_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if batch_idx % log_every == 0:
                avg = running_loss / (batch_idx + 1)
                print(f"epoch {epoch+1}/{epochs} | step {batch_idx}/{len(train_dl)} "
                      f"| loss {avg:.4f} | frac_masked {frac_masked:.2f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

        val_loss = evaluate(model, diffusion, val_dl, device, pad_token_id)
        print(f"== epoch {epoch+1} done in {time.time()-t0:.1f}s "
              f"| train_loss {running_loss/len(train_dl):.4f} | val_loss {val_loss:.4f} ==")

        if sample_every_epoch:
            generate_and_print(model, diffusion, tokenizer, device, n_samples=2, seq_len=max_length)

        torch.save({
            "model_state_dict": model.state_dict(), "epoch": epoch, "vocab_size": vocab_size,
            "mask_token_id": mask_token_id, "max_length": max_length, "schedule_name": schedule_name,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    return model, diffusion, tokenizer


@torch.no_grad()
def evaluate(model, diffusion, dataloader, device, pad_token_id):
    model.eval()
    total, n = 0.0, 0
    for x0 in dataloader:
        x0 = x0.to(device)
        attention_mask = (x0 != pad_token_id).long()
        loss, _ = diffusion.loss(model, x0, attention_mask=attention_mask)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def generate_and_print(model, diffusion, tokenizer, device, n_samples=2, seq_len=64):
    samples = diffusion.sample(model, batch_size=n_samples, seq_len=seq_len, device=device)
    print("---- generated samples (pure diffusion, full-sequence) ----")
    for i in range(n_samples):
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"[{i}] {text}")
    print("-------------------------------------------------------------")



# ======================================================================================
# 6. VISUALIZATION SCRIPT: How masked tokens become the answer
# ======================================================================================

def visualize_diffusion(
    model,
    diffusion: MaskedDiffusion,
    tokenizer,
    prompt: str,
    mask_ratio: float = 0.5,          # fraction of tokens to mask (choose the last part)
    show_every: int = 5,              # show state every N steps
    max_length: int = 64,
    device="cpu",
):
    """
    Demonstrates the reverse diffusion process on a given prompt.
    The prompt is tokenized, and a portion (the last `mask_ratio` of tokens)
    are replaced with [MASK]. Then we run the reverse process and print
    intermediate states to show how the masked tokens gradually get filled.
    """
    model.eval()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    if len(tokens) < max_length:
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
    x0 = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, L)

    # Determine which tokens to mask: we mask the last `mask_ratio` fraction
    L = x0.shape[1]
    num_mask = int(mask_ratio * L)
    # We'll mask from index (L - num_mask) to L-1
    mask_positions = torch.zeros(L, dtype=torch.bool)
    mask_positions[L - num_mask:] = True
    # For demonstration, we also mask any pad tokens, but let's keep them unmasked.
    # Actually, let's not mask the pad tokens (pad_token_id) because they are not meaningful.
    pad_id = tokenizer.pad_token_id
    non_pad = (x0[0] != pad_id)
    mask_positions = mask_positions & non_pad
    # Override: we only mask positions that are not pad
    # But we want to see the process on meaningful tokens, so we'll mask the last non-pad tokens.
    # Better: mask the last `num_mask` non-pad tokens.
    non_pad_indices = (x0[0] != pad_id).nonzero(as_tuple=True)[0]
    if len(non_pad_indices) > 0:
        # choose the last `num_mask` non-pad indices
        start_idx = max(0, len(non_pad_indices) - num_mask)
        mask_indices = non_pad_indices[start_idx:]
        mask_positions = torch.zeros(L, dtype=torch.bool)
        mask_positions[mask_indices] = True
    else:
        mask_positions = torch.zeros(L, dtype=torch.bool)

    # Create masked input x_t at t=1 (all masked) but we only want to mask those positions,
    # the others remain clean (as if we are conditioning on the unmasked part).
    # However, the forward process at t=1 masks everything; but for visualization,
    # we want to start from a state where the given tokens are clean and the rest are masked.
    # We can simulate this by setting x_t = x0, then force-mask the chosen positions.
    x_t = x0.clone()
    x_t[0, mask_positions] = diffusion.mask_token_id

    # We'll run the reverse process but only on the masked positions (the others stay fixed).
    # The standard sampling starts from all masked, but here we condition on the given prompt.
    # We'll adapt the sampling loop to keep the unmasked positions unchanged.
    T = diffusion.cfg.sampling_steps
    t_steps = torch.linspace(1.0, diffusion.cfg.eps, T + 1, device=device)

    print(f"Prompt: {prompt}")
    print(f"Masked positions (tokens): {tokenizer.decode(x_t[0].tolist(), skip_special_tokens=False)}")
    print("Starting reverse process...\n")

    def callback(step, t_cur, t_next, current_x):
        if step % show_every == 0 or step == T - 1:
            decoded = tokenizer.decode(current_x[0].tolist(), skip_special_tokens=True)
            print(f"Step {step+1}/{T} (t={t_cur.item():.3f} -> {t_next.item():.3f}):")
            print(f"  {decoded}\n")

    # We need to run a conditional sampling: the unmasked tokens stay as given.
    # We can simply run the normal sampling but we need to ensure that the unmasked
    # tokens are not altered. We can do this by resetting those positions after each step.
    # Or we can modify the sampling loop to condition on the fixed tokens.
    # We'll implement a custom loop here.

    # Start from x_t (with some masked positions)
    for step in range(T):
        t_cur = t_steps[step]
        t_next = t_steps[step + 1]
        t_batch = t_cur.expand(1)

        logits = model(x_t, t_batch, attention_mask=None)
        # We only need predictions for masked positions
        logits[..., diffusion.mask_token_id] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        x0_pred = torch.multinomial(
            probs.reshape(-1, diffusion.vocab_size), num_samples=1
        ).reshape(1, L)

        alpha_t = diffusion.schedule.alpha(t_cur).clamp(0.0, 1.0)
        alpha_s = diffusion.schedule.alpha(t_next).clamp(0.0, 1.0)
        denom = (1.0 - alpha_t).clamp(min=1e-8)
        p_reveal = ((alpha_s - alpha_t) / denom).clamp(0.0, 1.0)

        # Only reveal if currently masked
        is_masked = x_t.eq(diffusion.mask_token_id)
        rand = torch.rand(1, L, device=device)
        do_reveal = is_masked & (rand < p_reveal)

        x_t = torch.where(do_reveal, x0_pred, x_t)

        # Keep the originally unmasked tokens fixed (in case any were accidentally changed)
        # They should not have been masked, but we set them explicitly.
        x_t[0, ~mask_positions] = x0[0, ~mask_positions]

        if step % show_every == 0 or step == T - 1:
            decoded = tokenizer.decode(x_t[0].tolist(), skip_special_tokens=True)
            print(f"Step {step+1}/{T} (t={t_cur.item():.3f} -> {t_next.item():.3f}):")
            print(f"  {decoded}\n")

    # Final safety: fill any remaining masks with argmax
    still_masked = x_t.eq(diffusion.mask_token_id)
    if still_masked.any():
        t_final = torch.full((1,), diffusion.cfg.eps, device=device)
        logits = model(x_t, t_final)
        logits[..., diffusion.mask_token_id] = -float("inf")
        final_pred = logits.argmax(dim=-1)
        x_t = torch.where(still_masked, final_pred, x_t)

    final_text = tokenizer.decode(x_t[0].tolist(), skip_special_tokens=True)
    print("=== Final output ===")
    print(final_text)
    model.train()


# ======================================================================================
# 7. ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
        train(
        epochs=5,
        batch_size=16,
        max_length=128,
        lr=3e-4,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        sampling_steps=50,
        schedule_name="log_linear",   # or "cosine"
    )
    
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("flow_matching_dllm.pt", map_location=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    vocab_size = checkpoint["vocab_size"]
    mask_token_id = checkpoint["mask_token_id"]
    max_length = checkpoint["max_length"]
    model = MaskedDiffusionLM(
        vocab_size=vocab_size, mask_token_id=mask_token_id, max_length=max_length,
        d_model=checkpoint["d_model"], n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"], d_ff=checkpoint["d_ff"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    schedule_name = checkpoint.get("schedule_name", "log_linear")
    schedule = {"log_linear": LogLinearSchedule, "cosine": CosineSchedule}[schedule_name]()
    diffusion = MaskedDiffusion(mask_token_id, vocab_size, schedule, config=MDLMConfig(sampling_steps=50))
    
    # Now visualize with a question
    prompt = "The capital of France is"
    visualize_diffusion(model, diffusion, tokenizer, prompt, mask_ratio=0.5, show_every=5, max_length=64, device=device)

    print("To run visualization, uncomment the code in __main__ and provide a trained checkpoint.")