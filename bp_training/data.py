import os
import pytorch_lightning as pl
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class NERData(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        dataset = load_dataset("wikiann", "en")
        self.label_list = dataset["train"].features["ner_tags"].feature.names
        dataset = self.sample(dataset)
        dataset = dataset.map(self.tokenise_fn, batched=False)
        dataset = dataset.map(self.align_labels, batched=False)
        dataset = self.postprocess(dataset)
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=8,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            shuffle=True,
            batch_size=8,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"], batch_size=8, collate_fn=self.data_collator
        )

    @property
    def tokeniser(self):
        return AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    @property
    def data_collator(self):
        return DataCollatorForTokenClassification(tokenizer=self.tokeniser)

    def sample(self, data: DatasetDict) -> DatasetDict:
        return DatasetDict(
            {
                "train": data["train"].shuffle().select(range(8)),
                "validation": data["validation"].shuffle().select(range(8)),
                "test": data["test"].shuffle().select(range(8)),
            }
        )

    def tokenise_fn(self, data: DatasetDict):

        tokenised = self.tokeniser(
            data["tokens"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        tokenised["word_ids"] = tokenised.word_ids()

        return tokenised.data

    def align_labels(self, example):
        label = example["ner_tags"]
        label_ids = []
        previous_word_idx = None
        for word_idx in example["word_ids"]:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        example["labels"] = label_ids
        return example

    def postprocess(self, data: DatasetDict) -> DatasetDict:
        data = data.remove_columns(["tokens", "ner_tags", "langs", "spans", "word_ids"])
        data.set_format("torch")
        return data
