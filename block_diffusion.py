"""
BD3-LM: Block Discrete Denoising Diffusion Language Model (single file)
========================================================================
Reimplementation of the core algorithm from:
    Arriola, Gokaslan, Chiu, Yang, Qi, Han, Sahoo, Kuleshov -
    "Block Diffusion: Interpolating Between Autoregressive and Diffusion
    Language Models" (ICLR 2025 Oral)   https://github.com/kuleshov-group/bd3lms

--------------------------------------------------------------------------
THE CORE IDEA
--------------------------------------------------------------------------
Pure diffusion LMs (MDLM/SEDD) denoise the WHOLE sequence jointly and can't
easily generate beyond their training length. Pure AR LMs generate token by
token, left to right, with no parallelism within a token. BD3-LM sits in
between:

    1. Split the sequence into blocks of size L (e.g. L=4,8,16).
    2. Blocks are generated AUTOREGRESSIVELY, left to right (block b only
       ever conditions on blocks < b, which are fully clean/generated).
    3. WITHIN a block, generation is DISCRETE DIFFUSION (bidirectional,
       masked-absorbing-state denoising), so all L tokens in a block can be
       produced with fewer than L forward passes (parallel, not one at a
       time like AR), and unlike full-sequence diffusion, the number of
       diffusion steps stays fixed at L regardless of total sequence length,
       so arbitrary-length generation is possible.

Setting block_size=1 recovers a pure diffusion LM behaviour degenerates to
token-level AR-like updates; setting block_size=full sequence length
recovers a pure (non-block) diffusion LM. Block size is the AR<->diffusion
interpolation knob.

--------------------------------------------------------------------------
THE EFFICIENT TRAINING TRICK (the part that makes this non-trivial)
--------------------------------------------------------------------------
Naively, training would require a separate forward pass per block (b passes
for a sequence of b blocks) to get, for every block, "clean context from all
earlier blocks + noised current block". BD3-LM instead does this in **one**
forward pass by duplicating the input into two streams that are concatenated
along the sequence dimension and processed together by a *single* shared
transformer, using one custom attention mask:

    combined sequence = [ CLEAN stream (x)  |  NOISED stream (x_t) ]
                            length N              length N

Attention rules (query -> key allowed?):
  - CLEAN  -> CLEAN : allowed iff key's block <= query's block
                       (block-causal, bidirectional *within* a block -- this
                       stream only exists to produce KV context, so blocks
                       can see each other's clean tokens up to their own
                       block index; nothing here is used for a loss so there
                       is no leakage concern)
  - NOISED -> CLEAN : allowed iff key's block  < query's block (STRICT)
                       (the diffusion prediction for block b conditions on
                       fully clean PAST blocks only -- never its own block's
                       clean copy, or that would leak the answer)
  - NOISED -> NOISED: allowed iff key's block == query's block
                       (full bidirectional attention *within* the block
                       being denoised -- this is the "diffusion" part)
  - CLEAN  -> NOISED: never allowed
  - anything -> future block (key's block > query's block): never allowed

This single static (sequence-length-dependent, batch-independent) boolean
mask reproduces, in one forward/backward pass, exactly the conditioning
pattern needed to train every block's diffusion objective simultaneously --
this is the "efficient training algorithm" contribution of the paper.

--------------------------------------------------------------------------
NOISE SCHEDULE / LOSS (standard absorbing-state / MDLM convention)
--------------------------------------------------------------------------
t in [0, 1], t=0 -> clean, t=1 -> fully masked. Linear schedule:
    P(token masked | t) = t                      (masking probability)
    NELBO loss weight    = 1 / t                  (SUBS parameterization,
                                                    Sahoo et al. MDLM 2024)
Each BLOCK gets its own independently-sampled t_b (not a single global t for
the whole sequence) -- this is what lets a single forward pass realize many
different effective noise levels across blocks in one training step.

--------------------------------------------------------------------------
GENERATION
--------------------------------------------------------------------------
Blocks are generated strictly left to right. Within a block, an ancestral
/ Euler discrete-diffusion sampler runs a handful of steps from t=1 (all
[MASK]) down to t=0, unmasking tokens stochastically with probability
(t_cur - t_next) / t_cur per step (standard absorbing-state reverse
sampler), drawing revealed tokens from the model's categorical prediction.
Once a block is fully clean it becomes permanent left-context for all
future blocks -- exactly like AR decoding, but each "token" here is really
an L-token chunk produced by a few diffusion steps instead of one.
"""

import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


# ======================================================================================
# 1. DATALOADER (self-contained WikiText-2 loader, same pattern as before)
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
# 2. BLOCK-CAUSAL ATTENTION MASK  (the key structural ingredient)
# ======================================================================================

def build_block_diffusion_mask(seq_len, block_size, device):
    """
    Returns a (2*seq_len, 2*seq_len) BOOL mask where True = BLOCKED (cannot attend),
    matching torch.nn.MultiheadAttention's bool-mask convention.

    First `seq_len` positions = CLEAN stream, last `seq_len` positions = NOISED stream.
    See module docstring for the exact attention rules being encoded here.
    """
    assert seq_len % block_size == 0, "seq_len must be divisible by block_size"
    blk = torch.arange(seq_len, device=device) // block_size          # (N,)
    blk_combined = torch.cat([blk, blk])                                # (2N,)
    is_clean = torch.cat([
        torch.ones(seq_len, dtype=torch.bool, device=device),
        torch.zeros(seq_len, dtype=torch.bool, device=device),
    ])

    bi = blk_combined.unsqueeze(1)      # (2N, 1)  query block id
    bj = blk_combined.unsqueeze(0)      # (1, 2N)  key block id
    clean_i = is_clean.unsqueeze(1)
    clean_j = is_clean.unsqueeze(0)

    rule_clean_to_clean = clean_i & clean_j & (bj <= bi)            # block-causal, bidirectional in-block
    rule_noised_to_clean = (~clean_i) & clean_j & (bj < bi)         # strictly past clean blocks only
    rule_noised_to_noised = (~clean_i) & (~clean_j) & (bj == bi)    # full bidirectional within own block

    allowed = rule_clean_to_clean | rule_noised_to_clean | rule_noised_to_noised
    blocked = ~allowed
    return blocked  # (2N, 2N) bool, True = cannot attend


# ======================================================================================
# 3. MODEL: shared transformer over the [clean | noised] concatenated stream
# ======================================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Works on time tensors of arbitrary shape (..., ) -> (..., dim)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        orig_shape = t.shape
        t_flat = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t_flat[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.reshape(*orig_shape, self.dim)


class BlockDiffusionTransformerLayer(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero conditioning that varies PER TOKEN
    (needed because every token can carry a different block-time t)."""
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

    def forward(self, x, time_emb, attn_mask=None, key_padding_mask=None):
        # time_emb: (B, S, D) -- per-token conditioning
        scale1, shift1, scale2, shift2 = self.time_proj(time_emb).chunk(4, dim=-1)

        h = self.ln1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(
            h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out

        h = self.ln2(x) * (1 + scale2) + shift2
        x = x + self.ff(h)
        return x


class BD3LM(nn.Module):
    """
    Block Diffusion Language Model.
    forward() takes the CLEAN sequence and a (possibly partially-masked) NOISED
    sequence, runs the shared transformer once over the concatenated [clean|noised]
    stream with the block-diffusion attention mask, and returns logits for the
    NOISED stream positions only (that's the only stream we ever compute a loss /
    take predictions from).
    """
    def __init__(self, vocab_size, mask_token_id, seq_len, block_size,
                 d_model=384, n_heads=6, n_layers=6, d_ff=1536, dropout=0.1):
        super().__init__()
        assert seq_len % block_size == 0
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.seq_len = seq_len
        self.block_size = block_size
        self.num_blocks = seq_len // block_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            BlockDiffusionTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        # static structural mask, precomputed once per (seq_len, block_size)
        self.register_buffer(
            "structural_mask", build_block_diffusion_mask(seq_len, block_size, "cpu"), persistent=False
        )

    def forward(self, x_clean, x_noised, t_noised, attention_mask=None):
        """
        x_clean:  (B, N) ground-truth / already-generated tokens
        x_noised: (B, N) the block-diffusion input (masked at the block(s) being denoised)
        t_noised: (B, N) per-token diffusion time for the NOISED stream (0 for
                  positions that are irrelevant, e.g. during generation for blocks
                  not currently being sampled)
        attention_mask: (B, N) 1 = real token, 0 = padding
        returns logits: (B, N, V) predictions for the NOISED stream only
        """
        B, N = x_clean.shape
        device = x_clean.device
        assert N == self.seq_len, f"expected seq_len={self.seq_len}, got {N}"

        combined_tokens = torch.cat([x_clean, x_noised], dim=1)             # (B, 2N)
        t_clean = torch.zeros_like(t_noised)                                # clean stream: t=0 always
        t_combined = torch.cat([t_clean, t_noised], dim=1)                  # (B, 2N)

        h = self.token_emb(combined_tokens) + torch.cat([self.pos_emb, self.pos_emb], dim=1)
        t_emb = self.time_embed(t_combined)                                 # (B, 2N, D)

        key_padding_mask = None
        if attention_mask is not None:
            am = torch.cat([attention_mask, attention_mask], dim=1)
            key_padding_mask = (am == 0)

        attn_mask = self.structural_mask.to(device)

        for layer in self.layers:
            h = layer(h, t_emb, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        h = self.ln_out(h)
        logits_full = self.head(h)               # (B, 2N, V)
        logits_noised = logits_full[:, N:, :]     # only the noised-stream half matters
        return logits_noised


# ======================================================================================
# 4. NOISE SCHEDULE / CORRUPTION / LOSS  (per-block independent t, SUBS weighting)
# ======================================================================================

@dataclass
class BD3LMConfig:
    block_size: int = 16
    eps: float = 1e-3
    steps_per_block: int = 8   # diffusion steps used to denoise ONE block at sampling time


def corrupt_blockwise(x, block_size, mask_token_id, eps):
    """Samples an independent t_b ~ U(eps, 1) PER BLOCK, then masks each token in
    block b with probability t_b (linear schedule, t=0 clean / t=1 fully masked)."""
    B, N = x.shape
    num_blocks = N // block_size
    device = x.device

    t_block = torch.rand(B, num_blocks, device=device) * (1.0 - eps) + eps    # (B, num_blocks)
    t_noised = t_block.repeat_interleave(block_size, dim=1)                    # (B, N)

    rand = torch.rand(B, N, device=device)
    is_masked = rand < t_noised
    x_t = torch.where(is_masked, torch.full_like(x, mask_token_id), x)
    return x_t, is_masked, t_noised


def block_diffusion_loss(model, x, attention_mask, cfg: BD3LMConfig):
    """SUBS-parameterized NELBO: masked-position cross entropy weighted by 1/t,
    averaged over every block's independently-sampled noise level in one pass."""
    x_t, is_masked, t_noised = corrupt_blockwise(x, cfg.block_size, model.mask_token_id, cfg.eps)

    logits = model(x_clean=x, x_noised=x_t, t_noised=t_noised, attention_mask=attention_mask)

    loss_positions = is_masked
    if attention_mask is not None:
        loss_positions = loss_positions & attention_mask.bool()

    B, N = x.shape
    ce = F.cross_entropy(
        logits.reshape(-1, model.vocab_size), x.reshape(-1), reduction="none"
    ).reshape(B, N)

    weight = 1.0 / t_noised.clamp(min=cfg.eps)
    weighted_ce = ce * weight * loss_positions.float()
    denom = loss_positions.float().sum().clamp(min=1.0)
    loss = weighted_ce.sum() / denom

    return loss, is_masked.float().mean().item()


# ======================================================================================
# 5. BLOCK-AUTOREGRESSIVE GENERATION (diffusion within block, AR across blocks)
# ======================================================================================

@torch.no_grad()
def sample_bd3lm(model, batch_size, device, temperature=1.0, cfg: BD3LMConfig = BD3LMConfig()):
    model.eval()
    N, L = model.seq_len, model.block_size
    num_blocks = model.num_blocks
    mask_id = model.mask_token_id

    x = torch.full((batch_size, N), mask_id, dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, N, dtype=torch.long, device=device)

    for b in range(num_blocks):
        start, end = b * L, (b + 1) * L
        t_steps = torch.linspace(1.0, 0.0, cfg.steps_per_block + 1, device=device)

        for step in range(cfg.steps_per_block):
            t_cur, t_next = t_steps[step].item(), t_steps[step + 1].item()

            t_noised = torch.zeros(batch_size, N, device=device)
            t_noised[:, start:end] = t_cur

            # x_clean = x itself: blocks < b are already fully clean (used as real
            # context via the "noised -> clean, bj<bi" rule); block b and beyond are
            # still [MASK] in x, but the mask never lets any query see them, so this
            # is safe -- see build_block_diffusion_mask().
            logits = model(x_clean=x, x_noised=x, t_noised=t_noised, attention_mask=attention_mask)
            logits[:, start:end, mask_id] = -float("inf")  # never re-predict [MASK] itself

            probs = F.softmax(logits[:, start:end, :] / max(temperature, 1e-5), dim=-1)
            sampled = torch.multinomial(
                probs.reshape(-1, model.vocab_size), num_samples=1
            ).reshape(batch_size, L)

            block_tokens = x[:, start:end]
            is_masked = block_tokens.eq(mask_id)

            p_unmask = 1.0 if t_cur <= 1e-8 else (t_cur - t_next) / t_cur
            rand = torch.rand(batch_size, L, device=device)
            do_unmask = is_masked & (rand < p_unmask)

            x[:, start:end] = torch.where(do_unmask, sampled, block_tokens)

        # safety net: force-fill any residual [MASK] left in this block at t~0
        still_masked = x[:, start:end].eq(mask_id)
        if still_masked.any():
            t_noised = torch.zeros(batch_size, N, device=device)
            logits = model(x_clean=x, x_noised=x, t_noised=t_noised, attention_mask=attention_mask)
            logits[:, start:end, mask_id] = -float("inf")
            final_pred = logits[:, start:end, :].argmax(dim=-1)
            x[:, start:end] = torch.where(still_masked, final_pred, x[:, start:end])

    model.train()
    return x


# ======================================================================================
# 6. TRAINING LOOP
# ======================================================================================

def train(
    epochs=5, batch_size=16, max_length=128, block_size=16, lr=3e-4,
    d_model=384, n_heads=6, n_layers=6, d_ff=1536, steps_per_block=8,
    device=None, log_every=20, sample_every_epoch=True,
    checkpoint_path="bd3lm.pt",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, val_dl, tokenizer, mask_token_id, vocab_size = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    model = BD3LM(
        vocab_size=vocab_size, mask_token_id=mask_token_id, seq_len=max_length,
        block_size=block_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters | block_size={block_size} "
          f"| num_blocks={model.num_blocks}")

    cfg = BD3LMConfig(block_size=block_size, steps_per_block=steps_per_block)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05)

    pad_token_id = tokenizer.pad_token_id
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch_idx, x in enumerate(train_dl):
            x = x.to(device)
            attention_mask = (x != pad_token_id).long()

            loss, frac_masked = block_diffusion_loss(model, x, attention_mask, cfg)

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

        val_loss = evaluate(model, val_dl, device, pad_token_id, cfg)
        print(f"== epoch {epoch+1} done in {time.time()-t0:.1f}s "
              f"| train_loss {running_loss/len(train_dl):.4f} | val_loss {val_loss:.4f} ==")

        if sample_every_epoch:
            generate_and_print(model, tokenizer, device, n_samples=2, cfg=cfg)

        torch.save({
            "model_state_dict": model.state_dict(), "epoch": epoch, "vocab_size": vocab_size,
            "mask_token_id": mask_token_id, "seq_len": max_length, "block_size": block_size,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    return model, tokenizer, cfg


@torch.no_grad()
def evaluate(model, dataloader, device, pad_token_id, cfg):
    model.eval()
    total, n = 0.0, 0
    for x in dataloader:
        x = x.to(device)
        attention_mask = (x != pad_token_id).long()
        loss, _ = block_diffusion_loss(model, x, attention_mask, cfg)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def generate_and_print(model, tokenizer, device, n_samples=2, cfg: BD3LMConfig = BD3LMConfig()):
    samples = sample_bd3lm(model, batch_size=n_samples, device=device, cfg=cfg)
    print("---- generated samples (block-autoregressive) ----")
    for i in range(n_samples):
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"[{i}] {text}")
    print("----------------------------------------------------")


# ======================================================================================
# 7. ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=16,
        max_length=128,
        block_size=16,       # try 4 / 8 / 16 / 32 to move between diffusion-like and AR-like
        lr=3e-4,
        d_model=384,
        n_heads=8,
        n_layers=40,
        d_ff=1536,
        steps_per_block=8,
    )