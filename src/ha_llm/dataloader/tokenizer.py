from loguru import logger
from transformers import AutoTokenizer


def setup_tokenizer(name: str = "gpt2"):
    logger.debug("loading tokenizer {}", name)
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})
        logger.info("added [MASK] token to tokenizer {}", name)
    logger.debug("tokenizer {} vocab_size={}", name, len(tok))
    return tok
