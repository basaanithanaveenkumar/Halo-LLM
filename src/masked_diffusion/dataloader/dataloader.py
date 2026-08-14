import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class WikiTextDataset(Dataset):
    """
    This class handles loading the WikiText-2 dataset and preparing it for training.
    """
    def __init__(self, tokenizer_name="gpt2", max_length=128, split="train", size=None):
        # 1. Choose a tokenizer (we use GPT-2's by default)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-2 has no padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add mask token for diffusion, this is a special token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens(
                {"mask_token": "[MASK]"}
            )

        self.max_length = max_length

        # 2. Download the dataset from Hugging Face
        # 'wikitext' is the main dataset, 'wikitext-2-raw-v1' is the exact version
        #dataset = dataset = load_dataset("huggingface/wikitext", "wikitext-2-raw-v1", split=split)#load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = load_dataset(
                                "Salesforce/wikitext",
                                "wikitext-2-raw-v1",
                                split="train",
                            )
        # 3. If you want a smaller "toy" version, limit the number of samples
        if size is not None:
            dataset = dataset.select(range(min(size, len(dataset))))
        
        self.data = dataset

        # 4. Make sure the tokenizer has a [MASK] token (needed for diffusion)
        # if self.tokenizer.mask_token is None:
        #     self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.mask_token_id = self.tokenizer.mask_token_id

        print(f"Loaded {len(self.data)} samples from WikiText-2 ({split} split).")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the raw text for one sample
        item = self.data[idx]
        text = item['text']

        # Some entries in WikiText-2 are empty, so we fill them with a dummy token
        if not text or text.strip() == "":
            text = "[PAD]"

        # Convert text to numbers (token IDs), truncate if longer than max_length,
        # and pad with zeros if shorter.
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Return a 1D list of numbers (token IDs)
        input_ids = encoding['input_ids'].squeeze(0)
        return input_ids

# ------- Now we actually create the DataLoader -------

# Create a training set with only the first 2048 examples (that makes it a "toy")
train_dataset = WikiTextDataset(tokenizer_name="gpt2",split="train", size=2048)

# Create a validation set (to check progress during training)
val_dataset = WikiTextDataset(split="validation", size=512)

# The DataLoader will group samples into batches and shuffle them for training
batch_size = 16
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True   # ignore the last incomplete batch
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

# Let's test it – grab one batch and print its shape
sample_batch = next(iter(train_dataloader))
print(f"Sample batch shape: {sample_batch.shape}")   # This will be (16, 128)