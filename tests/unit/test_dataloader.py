import torch

from hale_llm.dataloader.dataset import sliding_windows


class _WordTok:
    pad_token_id = 0
    unk_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        words = text.split()
        out = []
        for w in words:
            out.append(2 + (sum(ord(c) for c in w) % 50))
        return out


def test_sliding_windows_stride_and_length():
    tok = _WordTok()
    text = " ".join(f"w{i}" for i in range(40))
    windows = sliding_windows(tok, text, max_length=8, stride_words=10)
    assert windows.ndim == 2
    assert windows.size(1) == 8
    assert windows.size(0) >= 4
    assert int(windows[1, 0]) == tok.encode("w10")[0]
    assert int(windows[0, 0]) == tok.encode("w0")[0]


def test_sliding_windows_short_text_pads():
    tok = _WordTok()
    windows = sliding_windows(tok, "hello world", max_length=8, stride_words=10)
    assert windows.shape == (1, 8)
    assert int(windows[0, -1]) == tok.pad_token_id


def test_sliding_windows_empty():
    tok = _WordTok()
    windows = sliding_windows(tok, "   ", max_length=4, stride_words=10)
    assert windows.shape == (0, 4)


def test_stride_words_must_be_positive():
    tok = _WordTok()
    try:
        sliding_windows(tok, "a b c", max_length=4, stride_words=0)
        assert False
    except ValueError:
        pass


def test_uses_overfit_from_source_or_text():
    from hale_llm.config.schema import RunConfig
    from hale_llm.dataloader.dataset import uses_overfit

    hf = RunConfig(variant="autoregressive", data={"source": "huggingface", "overfit_text": None})
    assert not uses_overfit(hf)
    text = RunConfig(variant="autoregressive", data={"overfit_text": "hello world"})
    assert uses_overfit(text)
    src = RunConfig(variant="autoregressive", data={"source": "overfit", "overfit_text": "hello"})
    assert uses_overfit(src)


def test_passage_prompt_takes_prefix():
    from hale_llm.config.schema import RunConfig
    from hale_llm.dataloader.dataset import _is_heading, dataset_prompt, passage_prompt

    assert _is_heading("= Game =")
    assert not _is_heading("The game was released in 1991.")
    assert passage_prompt("one two three four five", 3) == "one two three"
    cfg = RunConfig(variant="autoregressive", viz={"prompt_source": "dataset", "prompt_words": 2})
    passages = ["alpha beta gamma", "delta epsilon zeta"]
    assert dataset_prompt(cfg, seed=0, passages=passages) == "alpha beta"
    assert dataset_prompt(cfg, seed=1, passages=passages) == "delta epsilon"
