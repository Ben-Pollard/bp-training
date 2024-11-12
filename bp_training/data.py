"""
Adapters for huggingface datasets for use with Lightning
"""

import os

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class NERData(LightningDataModule):
    """
    A PyTorch Lightning DataModule for Named Entity Recognition (NER) tasks
    using the WikiANN dataset.
    """

    def __init__(self) -> None:
        """
        Initializes the NERData module by loading and processing the WikiANN dataset.
        """
        super().__init__()
        dataset = load_dataset("wikiann", "en")
        self.label_list = dataset["train"].features["ner_tags"].feature.names  # type: ignore
        dataset = self.sample(dataset)  # type: ignore
        dataset = dataset.map(self.tokenise_fn, batched=False)
        dataset = dataset.map(self.align_labels, batched=False)
        dataset = self.postprocess(dataset)
        self.dataset = dataset

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.
        """
        return DataLoader(
            self.dataset["train"],  # type: ignore
            shuffle=True,
            batch_size=8,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.
        """
        return DataLoader(
            self.dataset["validation"],  # type: ignore
            batch_size=8,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.
        """
        return DataLoader(
            self.dataset["test"], batch_size=8, collate_fn=self.data_collator  # type: ignore
        )

    @property
    def tokeniser(self):
        """
        Returns the tokenizer for token classification tasks.
        """
        return AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    @property
    def data_collator(self):
        """
        Returns the data collator for token classification tasks.
        """
        return DataCollatorForTokenClassification(tokenizer=self.tokeniser)

    def sample(self, data: DatasetDict) -> DatasetDict:
        """
        Samples a subset of the dataset for training, validation, and testing.

        Args:
            data (DatasetDict): The original dataset.

        Returns:
            DatasetDict: A sampled subset of the dataset.
        """
        return DatasetDict(
            {
                "train": data["train"].shuffle().select(range(8)),
                "validation": data["validation"].shuffle().select(range(8)),
                "test": data["test"].shuffle().select(range(8)),
            }
        )

    def tokenise_fn(self, data: DatasetDict):
        """
        Tokenizes the input data using the tokenizer.

        Args:
            data (DatasetDict): The input data to tokenize.

        Returns:
            dict: The tokenized data.
        """
        tokenised = self.tokeniser(
            data["tokens"],  # type: ignore
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        tokenised["word_ids"] = tokenised.word_ids()

        return tokenised.data

    def align_labels(self, example):
        """
        Aligns the labels with the tokenized inputs.

        Args:
            example (dict): A single example containing 'ner_tags' and 'word_ids'.

        Returns:
            dict: The example with aligned labels.
        """
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
        """
        Post-processes the dataset by removing unnecessary columns
          and setting the format to PyTorch.

        Args:
            data (DatasetDict): The dataset to post-process.

        Returns:
            DatasetDict: The post-processed dataset.
        """
        data = data.remove_columns(["tokens", "ner_tags", "langs", "spans", "word_ids"])
        data.set_format("torch")
        return data
