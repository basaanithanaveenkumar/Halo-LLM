"""Same drawing path as `visualize_unmasking` in flowmatching.py.

Prompt = steelblue, already generated answer = orange, [MASK] = dimgray.
Current decoding (AR / MTP / active block) = black.
Block diffusion also draws a light overlay box (no border) on the active block.
"""

from __future__ import annotations

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from loguru import logger
from matplotlib.patches import Rectangle

CURRENT_COLOR = "black"
HIGHLIGHT_VARIANTS = {"autoregressive", "mtp", "block_diffusion"}
OVERLAY_VARIANTS = {"block_diffusion"}
OVERLAY_FACE = "#FFF3C4"
OVERLAY_ALPHA = 0.55
OVERLAY_HEIGHT = 0.58
OVERLAY_PAD_X = 0.18
LABEL_FONTSIZE = 12
TOKEN_FONTSIZE = 13
TITLE_FONTSIZE = 14


def compute_positions(tokens, char_spacing=0.5):
    x, positions = 0.0, []
    for tok in tokens:
        positions.append(x + len(tok) / 2.0)
        x += len(tok) + char_spacing
    return positions, x


def wrap_tokens(tokens, max_width, char_spacing=0.5):
    lines = []
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


def _current_indices(variant, ids, prev_ids, prompt_len, meta, mask_id):
    if variant not in HIGHLIGHT_VARIANTS:
        return set()
    n = len(ids)
    if variant == "block_diffusion":
        bs = int(meta.get("block_size") or 0)
        b_idx = int(meta.get("block", -1))
        if bs <= 0 or b_idx < 0:
            return set()
        start, end = b_idx * bs, min((b_idx + 1) * bs, n)
        return {i for i in range(max(start, prompt_len), end)}
    active = set()
    for i in range(prompt_len, n):
        if mask_id is not None and int(ids[i]) == mask_id:
            continue
        if prev_ids is None or i >= len(prev_ids) or int(prev_ids[i]) != int(ids[i]):
            active.add(i)
    return active


def _overlay_runs(line, L_prompt, current, answer_tokens):
    runs = []
    run = []
    for orig_idx, x_pos in line:
        abs_i = L_prompt + orig_idx
        if abs_i in current:
            run.append((answer_tokens[orig_idx], x_pos))
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def render_unmasking_gif(
    snapshots,
    L_prompt,
    mask_token_id,
    output_gif="diffusion_reveal.gif",
    fps=3,
    dpi=120,
    max_line_width=100.0,
    row_spacing=0.8,
    overlay_current=False,
):
    """Literal port of flowmatching.visualize_unmasking drawing, plus optional current-set."""
    max_answer_lines = 1
    for s in snapshots:
        answer_tokens = s[1][L_prompt:]
        lines = wrap_tokens(answer_tokens, max_line_width)
        max_answer_lines = max(max_answer_lines, len(lines) if lines else 1)

    canvas_width = max_line_width + 6.0
    canvas_height = 2.0 + (max_answer_lines * row_spacing)

    fig, ax = plt.subplots(figsize=(canvas_width * 0.2 + 2.0, canvas_height))
    fig.patch.set_facecolor("white")

    def draw_frame(fi):
        ax.clear()
        ax.axis("off")

        y_top = 0.5 * (max_answer_lines * row_spacing)
        ax.set_xlim(-3, max_line_width + 2)
        ax.set_ylim(-y_top - row_spacing, y_top + row_spacing)

        t_val, tokens_str, seq_ids, current = snapshots[fi]
        seq_ids = np.asarray(seq_ids)

        ax.text(-1, y_top, "Prompt:", ha="right", va="center", fontsize=LABEL_FONTSIZE, weight="bold")
        for i, tok in enumerate(tokens_str[:L_prompt]):
            pos = compute_positions(tokens_str[:L_prompt])[0][i]
            ax.text(
                pos,
                y_top,
                tok,
                ha="center",
                va="center",
                fontsize=TOKEN_FONTSIZE,
                color="steelblue",
                weight="bold",
                family="monospace",
            )

        answer_tokens = tokens_str[L_prompt:]
        wrapped_lines = wrap_tokens(answer_tokens, max_line_width)
        ax.text(-1, y_top - row_spacing, "Answer:", ha="right", va="center", fontsize=LABEL_FONTSIZE, weight="bold")

        for line_idx, line in enumerate(wrapped_lines):
            y_pos = y_top - ((line_idx + 1) * row_spacing)
            if overlay_current and current:
                for run in _overlay_runs(line, L_prompt, current, answer_tokens):
                    left = run[0][1] - len(run[0][0]) / 2.0
                    right = run[-1][1] + len(run[-1][0]) / 2.0
                    ax.add_patch(
                        Rectangle(
                            (left - OVERLAY_PAD_X, y_pos - OVERLAY_HEIGHT / 2.0),
                            (right - left) + 2.0 * OVERLAY_PAD_X,
                            OVERLAY_HEIGHT,
                            facecolor=OVERLAY_FACE,
                            edgecolor="none",
                            linewidth=0,
                            alpha=OVERLAY_ALPHA,
                            zorder=0,
                        )
                    )
            for orig_idx, x_pos in line:
                tok = answer_tokens[orig_idx]
                abs_i = L_prompt + orig_idx
                is_m = mask_token_id is not None and int(seq_ids[abs_i]) == mask_token_id
                if is_m:
                    color = "dimgray"
                elif abs_i in current:
                    color = CURRENT_COLOR
                else:
                    color = "orange"
                ax.text(
                    x_pos,
                    y_pos,
                    tok,
                    ha="center",
                    va="center",
                    fontsize=TOKEN_FONTSIZE,
                    color=color,
                    weight="bold",
                    family="monospace",
                    zorder=1,
                )

        ax.set_title(
            f"t = {t_val:.2f}  (frame {fi + 1}/{len(snapshots)})",
            fontsize=TITLE_FONTSIZE,
            y=-0.15,
            pad=20,
        )

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=200, repeat=False)
    anim.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    logger.info("GIF saved as {}", output_gif)
    print(f"GIF saved as {output_gif}")
    return output_gif


def visualize_generation(
    model,
    tokenizer,
    sampler,
    prompt="The Sinclair Scientific Programmable was introduced in ",
    device="cpu",
    max_new_tokens=50,
    sampling_steps=24,
    temperature=1.0,
    output_gif="diffusion_reveal.gif",
    fps=3,
    dpi=120,
    max_line_width=100.0,
    row_spacing=0.8,
    variant="flow_matching",
    **_,
):
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"].to(device)
    L_prompt = prompt_ids.shape[1]

    final_seq, history = sampler(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        sampling_steps=sampling_steps,
        return_history=True,
    )

    mask_id = tokenizer.mask_token_id
    snapshots = []
    prev_ids = None
    for item in history:
        if len(item) == 3:
            t_val, seq, meta = item
            meta = dict(meta)
        else:
            t_val, seq = item
            meta = {}
        ids = seq[0].cpu().numpy() if torch.is_tensor(seq) else np.asarray(seq)
        seq_list = ids.tolist()
        tokens_str = ["[MASK]" if tid == mask_id else tokenizer.decode([tid]) for tid in seq_list]
        current = _current_indices(variant, ids, prev_ids, L_prompt, meta, mask_id)
        snapshots.append((float(t_val), tokens_str, ids, current))
        prev_ids = ids

    render_unmasking_gif(
        snapshots,
        L_prompt,
        mask_id,
        output_gif=output_gif,
        fps=fps,
        dpi=dpi,
        max_line_width=max_line_width,
        row_spacing=row_spacing,
        overlay_current=variant in OVERLAY_VARIANTS,
    )
    print("\nFinal generated text:")
    print(tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True))
    logger.info("generated: {}", tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True))
    return final_seq


def _token_strings(tokenizer, ids: torch.Tensor) -> list[str]:
    mask_id = getattr(tokenizer, "mask_token_id", None)
    out = []
    for tid in ids.tolist():
        if mask_id is not None and tid == mask_id:
            out.append("[MASK]")
        else:
            out.append(tokenizer.decode([tid]))
    return out


def render_token_grid_gif(
    snapshots,
    tokenizer,
    prompt_len,
    output_gif="diffusion_reveal_grid.gif",
    cols=10,
    cell_size=0.8,
    fps=3,
    dpi=120,
    title_prefix="hale-llm",
):
    from matplotlib.patches import Rectangle

    total_len = len(snapshots[0][1])
    rows = int(np.ceil(total_len / cols))
    pad_len = rows * cols - total_len
    padded = [""] * pad_len
    fig, ax = plt.subplots(figsize=(cols * cell_size + 2.0, rows * cell_size + 2.0))
    fig.patch.set_facecolor("white")

    def draw_frame(fi):
        ax.clear()
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        t_val, tokens_str, seq_ids = snapshots[fi][:3]
        full_tokens = tokens_str + padded
        full_ids = list(np.asarray(seq_ids)) + [tokenizer.pad_token_id] * pad_len
        mask_id = getattr(tokenizer, "mask_token_id", None)
        for idx, (tok, tid) in enumerate(zip(full_tokens, full_ids)):
            if tok == "":
                continue
            row, col = divmod(idx, cols)
            x, y = col, rows - 1 - row
            is_prompt = idx < prompt_len
            is_mask = mask_id is not None and tid == mask_id
            if is_prompt:
                text_color = "steelblue"
            elif is_mask:
                ax.add_patch(Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor="#EEEEEE", edgecolor="#CCCCCC"))
                text_color = "dimgray"
            else:
                text_color = "orange"
            ax.text(x, y, tok, ha="center", va="center", fontsize=12, color=text_color, weight="bold", family="monospace")
        ax.set_title(f"{title_prefix}  t={t_val:.2f}  ({fi + 1}/{len(snapshots)})", fontsize=14, pad=16)

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=200, repeat=False)
    anim.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return output_gif
