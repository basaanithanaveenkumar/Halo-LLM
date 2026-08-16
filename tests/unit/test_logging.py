from ha_llm.utils.logging import setup_logging
from ha_llm.utils.tensorboard import log_scalars, make_writer


def test_setup_logging_levels(tmp_path):
    log_file = tmp_path / "ha.log"
    setup_logging(level="DEBUG", log_file=str(log_file), force=True)
    from loguru import logger

    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    text = log_file.read_text()
    assert "d" in text and "i" in text and "w" in text and "e" in text


def test_tensorboard_writer(tmp_path):
    writer = make_writer(str(tmp_path), enabled=True, run_name="test")
    assert writer is not None
    log_scalars(writer, 0, {"train/loss": 1.23})
    writer.close()
    events = list((tmp_path / "test").glob("events.out.tfevents.*"))
    assert events
