"""Stacked generation snapshots. Highlight only revealed tokens, never mask runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.patches import FancyBboxPatch

BG = "#F7F6F2"
TEXT_COLOR = "#2C3338"
PROMPT_COLOR = "#5C6B73"
MASK_COLOR = "#C5C8C6"
HIGHLIGHT = "#FFE08A"
HIGHLIGHT_EDGE = "#E0B94A"
STEP_COLOR = "#9AA3A7"

HIGHLIGHT_MODE = {
    "autoregressive": "new_tokens",
    "mtp": "new_tokens",
    "diffusion": "new_tokens",
    "flow_matching": "new_tokens",
    "block_diffusion": "new_tokens",
}

MASK_GLYPH = "·"
CHAR_W = 0.0076
FONTSIZE = 10
MAX_X = 0.98
ROW_WRAP_DY = 0.11


@dataclass
class TokenSpan:
    text: str
    highlight: bool = False
    muted: bool = False
    prompt: bool = False


@dataclass
class StepRow:
    tokens: list[TokenSpan] = field(default_factory=list)
    # kept so older tests that pass text= still construct
    text: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)


def _highlight_mode(variant: str) -> str:
    return HIGHLIGHT_MODE.get(variant, "new_tokens")


def _token_piece(tokenizer, tid: int, mask_id, pad_id) -> tuple[str, str] | None:
    """Return (display_text, role) or None to skip."""
    if pad_id is not None and tid == pad_id:
        return None
    if mask_id is not None and tid == mask_id:
        return MASK_GLYPH, "mask"
    piece = tokenizer.decode([tid])
    if not piece:
        return None
    return piece, "tok"


def _n_revealed(ids: np.ndarray, mask_id, prompt_len: int) -> int:
    if mask_id is None:
        return max(len(ids) - prompt_len, 0)
    return int(np.sum(np.asarray(ids)[prompt_len:] != int(mask_id)))


def _pick_indices(unpacked: list, tokenizer, prompt_len: int, n_show: int) -> list[int]:
    mask_id = getattr(tokenizer, "mask_token_id", None)
    scores = [_n_revealed(ids, mask_id, prompt_len) for ids, _ in unpacked]
    # keep first, last, and frames where reveal count jumps
    jumps = [0]
    last = scores[0]
    for i, s in enumerate(scores[1:], start=1):
        if s > last:
            jumps.append(i)
            last = s
    if jumps[-1] != len(unpacked) - 1:
        jumps.append(len(unpacked) - 1)
    if len(jumps) <= n_show:
        return jumps
    sel = np.unique(np.linspace(0, len(jumps) - 1, n_show, dtype=int))
    return [jumps[i] for i in sel]


def history_to_rows(
    history: list,
    tokenizer,
    prompt_len: int,
    variant: str,
    n_show: int = 4,
) -> list[StepRow]:
    unpacked: list[tuple[np.ndarray, dict]] = []
    for item in history:
        if len(item) == 3:
            _t, seq, meta = item
        else:
            _t, seq = item
            meta = {}
        row = seq[0].detach().cpu().numpy() if torch.is_tensor(seq) else np.asarray(seq)
        unpacked.append((row, dict(meta)))
    if not unpacked:
        return []
    idxs = _pick_indices(unpacked, tokenizer, prompt_len, n_show)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    rows: list[StepRow] = []
    for i in idxs:
        ids, _meta = unpacked[i]
        prev = unpacked[i - 1][0] if i > 0 else None
        tokens: list[TokenSpan] = []
        for t_i, raw in enumerate(np.asarray(ids).tolist()):
            parsed = _token_piece(tokenizer, int(raw), mask_id, pad_id)
            if parsed is None:
                continue
            piece, role = parsed
            is_prompt = t_i < prompt_len
            newly = False
            if not is_prompt and role != "mask":
                if prev is None or t_i >= len(prev) or int(prev[t_i]) != int(raw):
                    newly = True
                elif mask_id is not None and int(prev[t_i]) == int(mask_id):
                    newly = True
            tokens.append(
                TokenSpan(
                    text=piece,
                    highlight=newly,
                    muted=role == "mask",
                    prompt=is_prompt,
                )
            )
        rows.append(StepRow(tokens=tokens))
    return rows


def _draw_static(ax, rows: list[StepRow], n_visible: int | None = None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)
    n_visible = len(rows) if n_visible is None else n_visible
    n_visible = max(1, min(n_visible, len(rows)))
    n = max(len(rows), 1)
    for i, row in enumerate(rows[:n_visible]):
        y = 0.86 - i * (0.70 / max(n - 1, 1) if n > 1 else 0.0)
        if n == 1:
            y = 0.50
        ax.text(0.015, y, str(i + 1), fontsize=9, color=STEP_COLOR, va="center", ha="left")
        tokens = row.tokens
        if not tokens and row.text:
            tokens = [TokenSpan(row.text)]
        _draw_tokens(ax, tokens, x0=0.05, y=y)


def _draw_tokens(ax, tokens: list[TokenSpan], x0: float, y: float) -> None:
    x = x0
    line_y = y
    for tok in tokens:
        w = CHAR_W * max(len(tok.text), 1)
        if x + w > MAX_X and x > x0:
            line_y -= ROW_WRAP_DY
            x = x0
        if tok.muted:
            ax.text(x, line_y, tok.text, fontsize=FONTSIZE, color=MASK_COLOR, va="center", ha="left", family="monospace")
        else:
            color = PROMPT_COLOR if tok.prompt else TEXT_COLOR
            if tok.highlight:
                ax.add_patch(
                    FancyBboxPatch(
                        (x - 0.001, line_y - 0.035),
                        w + 0.002,
                        0.07,
                        boxstyle="round,pad=0.004,rounding_size=0.008",
                        facecolor=HIGHLIGHT,
                        edgecolor=HIGHLIGHT_EDGE,
                        linewidth=0.4,
                    )
                )
            ax.text(
                x,
                line_y,
                tok.text,
                fontsize=FONTSIZE,
                color=color,
                va="center",
                ha="left",
                family="monospace",
            )
        x += w


def render_paper_figure(
    rows: list[StepRow],
    variant: str,
    output_path: str,
    *,
    dpi: int = 140,
    fps: int = 2,
    as_gif: bool = True,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nchars = max(
        (sum(len(t.text) for t in r.tokens) if r.tokens else len(r.text) for r in rows),
        default=40,
    )
    wraps = max(1, int(np.ceil(nchars * CHAR_W / (MAX_X - 0.05))))
    fig_w = 11.0
    fig_h = 1.15 + 0.55 * max(len(rows), 1) * max(wraps, 1)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 8.5)))
    fig.patch.set_facecolor(BG)

    def draw_frame(fi: int) -> None:
        ax.clear()
        _draw_static(ax, rows, n_visible=fi + 1)

    gif_path = output_path if output_path.lower().endswith(".gif") else output_path.rsplit(".", 1)[0] + ".gif"
    png_path = output_path.rsplit(".", 1)[0] + ".png"

    if as_gif:
        anim = animation.FuncAnimation(
            fig, draw_frame, frames=max(len(rows), 1), interval=800, repeat=True
        )
        anim.save(gif_path, writer="pillow", fps=fps, dpi=dpi)

    ax.clear()
    _draw_static(ax, rows)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("paper-style viz saved {} and {}", gif_path, png_path)
    return gif_path


def visualize_paper_style(
    model,
    tokenizer,
    sampler,
    prompt: str,
    device: str,
    variant: str,
    max_new_tokens: int = 32,
    sampling_steps: int = 32,
    temperature: float = 1.0,
    output_gif: str = "generation.gif",
    n_show: int = 4,
    fps: int = 2,
) -> torch.Tensor:
    model.eval()
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=getattr(getattr(model, "backbone", None), "max_length", 64),
    )["input_ids"].to(device)
    prompt_len = int(prompt_ids.shape[1])
    out = sampler(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        sampling_steps=sampling_steps,
        return_history=True,
    )
    final_seq, history = out
    rows = history_to_rows(history, tokenizer, prompt_len, variant, n_show=n_show)
    if not rows:
        logger.warning("sampler returned no history; skipping viz")
        return final_seq
    render_paper_figure(rows, variant, output_gif, fps=fps, as_gif=True)
    logger.info("generated: {}", tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True))
    return final_seq
