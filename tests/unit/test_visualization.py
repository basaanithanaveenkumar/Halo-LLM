import numpy as np

from hale_llm.visualization.block_diffusion import render_block_diffusion_gif
from hale_llm.visualization.unmasking import render_token_grid_gif, render_unmasking_gif


class _Tok:
    pad_token_id = 0
    mask_token_id = 1

    def decode(self, ids):
        if isinstance(ids, list) and len(ids) == 1:
            return {1: "[MASK]", 2: "hi", 3: "there"}.get(ids[0], "?")
        return " ".join(self.decode([i]) for i in ids)


def test_render_token_grid_gif(tmp_path):
    tok = _Tok()
    ids = np.array([2, 1, 1, 3])
    snaps = [
        (0.0, ["hi", "[MASK]", "[MASK]", "there"], ids),
        (1.0, ["hi", "hi", "there", "there"], np.array([2, 2, 3, 3])),
    ]
    out = tmp_path / "t.gif"
    render_token_grid_gif(snaps, tok, prompt_len=1, output_gif=str(out), cols=2, fps=2)
    assert out.exists() and out.stat().st_size > 0


def test_render_block_diffusion_gif(tmp_path):
    tok = _Tok()
    snaps = [
        (
            1.0,
            ["hi", "[MASK]", "[MASK]", "[MASK]"],
            np.array([2, 1, 1, 1]),
            {"block": -1, "num_blocks": 2, "block_size": 2},
        ),
        (
            0.5,
            ["hi", "there", "[MASK]", "[MASK]"],
            np.array([2, 3, 1, 1]),
            {"block": 0, "num_blocks": 2, "block_size": 2},
        ),
        (
            0.0,
            ["hi", "there", "hi", "there"],
            np.array([2, 3, 2, 3]),
            {"block": 1, "num_blocks": 2, "block_size": 2},
        ),
    ]
    out = tmp_path / "bd.gif"
    render_block_diffusion_gif(snaps, tok, prompt_len=1, block_size=2, output_gif=str(out), fps=2)
    assert out.exists() and out.stat().st_size > 0


def test_render_unmasking_gif(tmp_path):
    snaps = [
        (0.0, ["hello", "[MASK]", "[MASK]"], np.array([2, 1, 1]), set()),
        (0.5, ["hello", "there", "[MASK]"], np.array([2, 3, 1]), {1}),
        (1.0, ["hello", "there", "hi"], np.array([2, 3, 2]), {2}),
    ]
    out = tmp_path / "u.gif"
    render_unmasking_gif(snaps, L_prompt=1, mask_token_id=1, output_gif=str(out), fps=2)
    assert out.exists() and out.stat().st_size > 0
    overlay = tmp_path / "overlay.gif"
    render_unmasking_gif(
        snaps,
        L_prompt=1,
        mask_token_id=1,
        output_gif=str(overlay),
        fps=2,
        overlay_current=True,
    )
    assert overlay.exists() and overlay.stat().st_size > 0
