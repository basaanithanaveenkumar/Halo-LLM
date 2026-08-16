"""Block-diffusion GIF: one row per block, tokens unmask left-to-right across blocks.

Does not import models_block_diffusion — uses the registered sampler + history metadata.
"""

from __future__ import annotations

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from loguru import logger

from ha_llm.visualization.unmasking import _token_strings


def render_block_diffusion_gif(
    snapshots: list[tuple[float, list[str], np.ndarray, dict]],
    tokenizer,
    prompt_len: int,
    block_size: int,
    output_gif: str = "block_diffusion_reveal.gif",
    cell_size: float = 0.85,
    fps: int = 4,
    dpi: int = 120,
) -> str:
    """Each row is one block. Active block is highlighted while it denoises."""
    tokens0 = snapshots[0][1]
    seq_len = len(tokens0)
    num_blocks = int(np.ceil(seq_len / block_size))
    cols = block_size
    rows = num_blocks

    fig_w = cols * cell_size + 2.8
    fig_h = rows * cell_size + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    mask_id = getattr(tokenizer, "mask_token_id", None)

    def draw_frame(fi):
        ax.clear()
        ax.set_xlim(-2.2, cols - 0.3)
        ax.set_ylim(-0.6, rows - 0.3)
        ax.set_aspect("equal")
        ax.axis("off")

        t_val, tokens_str, seq_ids, meta = snapshots[fi]
        active = int(meta.get("block", -1))

        for idx, (tok, tid) in enumerate(zip(tokens_str, seq_ids)):
            b_idx = idx // block_size
            col = idx % block_size
            x, y = col, rows - 1 - b_idx
            is_prompt = idx < prompt_len
            is_mask = mask_id is not None and int(tid) == mask_id
            is_active = b_idx == active
            is_done = 0 <= b_idx < active or (b_idx == active and not is_mask and not is_prompt)

            if col == 0:
                ax.text(
                    -1.7,
                    y,
                    f"b{b_idx}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="#1F618D" if is_active else "#7F8C8D",
                    weight="bold" if is_active else "normal",
                    family="sans-serif",
                )

            if is_prompt:
                text_color, weight = "darkblue", "bold"
            elif is_mask:
                face = "#F9E79F" if is_active else "#D6EAF8"
                edge = "#B7950B" if is_active else "#A9CCE3"
                ax.add_patch(
                    Rectangle(
                        (x - 0.4, y - 0.4),
                        0.8,
                        0.8,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=1.2 if is_active else 1,
                    )
                )
                text_color, weight = ("#7D6608" if is_active else "gray"), "normal"
            elif is_active or is_done:
                if is_active:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.4, y - 0.4),
                            0.8,
                            0.8,
                            facecolor="#FDEBD0",
                            edgecolor="#E67E22",
                            linewidth=1.2,
                        )
                    )
                    text_color, weight = "orange", "bold"
                else:
                    text_color, weight = "steelblue", "bold"
            else:
                text_color, weight = "dimgray", "normal"

            ax.text(
                x,
                y,
                tok,
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                weight=weight,
                family="sans-serif",
            )

        block_label = "prompt" if active < 0 else f"block {active + 1}/{num_blocks}"
        ax.set_title(
            f"Block diffusion  {block_label}  t={t_val:.2f}  "
            f"({fi + 1}/{len(snapshots)})",
            fontsize=13,
            pad=14,
        )

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(snapshots), interval=200, repeat=True
    )
    anim.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    logger.info("GIF saved as {}", output_gif)
    return output_gif


def visualize_block_diffusion(
    model,
    tokenizer,
    sampler,
    prompt: str,
    device: str,
    max_new_tokens: int = 32,
    sampling_steps: int | None = None,
    temperature: float = 1.0,
    output_gif: str = "block_diffusion_reveal.gif",
    fps: int = 4,
) -> torch.Tensor:
    """Run the registered block-diffusion sampler and write a per-block reveal GIF."""
    model.eval()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    prompt_len = prompt_ids.shape[1]
    block_size = getattr(model, "block_size", None) or model.backbone.block_size
    steps = sampling_steps or getattr(model, "steps_per_block", 8)

    final_seq, history = sampler(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        sampling_steps=steps,
        return_history=True,
    )

    snapshots = []
    for item in history:
        if len(item) == 3:
            t_val, seq, meta = item
        else:
            t_val, seq = item
            meta = {"block": -1, "block_size": block_size}
        row = seq[0].detach().cpu()
        snapshots.append(
            (float(t_val), _token_strings(tokenizer, row), row.numpy(), meta)
        )

    render_block_diffusion_gif(
        snapshots,
        tokenizer,
        prompt_len,
        block_size=block_size,
        output_gif=output_gif,
        fps=fps,
    )
    logger.info("generated: {}", tokenizer.decode(final_seq[0].tolist(), skip_special_tokens=True))
    return final_seq
