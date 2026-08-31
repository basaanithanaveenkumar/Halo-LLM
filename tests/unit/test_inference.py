import hale_llm.models  # noqa: F401
from hale_llm.config.schema import RunConfig
from hale_llm.core.registry import get_variant
from hale_llm.inference import session_from_model


def test_session_generate_returns_text(tiny_tokenizer):
    cfg = RunConfig(
        variant="autoregressive",
        train={"batch_size": 2, "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        sample={"max_new_tokens": 3, "prompt": "hi", "temperature": 1.0},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    session = session_from_model(model, tiny_tokenizer, cfg, device="cpu")
    result = session.generate("hi")
    assert result.prompt == "hi"
    assert result.token_ids.ndim == 2
    assert result.token_ids.size(1) > 0
    assert isinstance(result.text, str)
    hist = session.generate("hi", return_history=True)
    assert hist.history is not None
    assert len(hist.history) >= 1


def test_load_session_falls_back_when_checkpoint_missing(tiny_tokenizer, tmp_path):
    from hale_llm.inference import load_session

    cfg = RunConfig(
        variant="autoregressive",
        train={"checkpoint_path": str(tmp_path / "missing.pt"), "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=False)
    assert session.model is not None
    result = session.generate("hi")
    assert isinstance(result.text, str)


def test_load_session_explicit_missing_checkpoint_raises(tiny_tokenizer, tmp_path):
    from hale_llm.inference import load_session
    import pytest

    cfg = RunConfig(
        variant="autoregressive",
        train={"checkpoint_path": str(tmp_path / "also_missing.pt"), "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    with pytest.raises(FileNotFoundError):
        load_session(cfg, str(tmp_path / "explicit.pt"), tokenizer=tiny_tokenizer)
