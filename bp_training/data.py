import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class YelpReviewBertData:
    def __init__(self) -> None:
        dataset = load_dataset("yelp_review_full")
        dataset = self.sample(dataset)
        dataset = dataset.map(self.tokenise_fn, batched=True)
        dataset = self.postprocess(dataset)
        self.dataset = dataset

    @property
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=8)

    @property
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=8)

    @property
    def tokeniser(self):
        return AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def sample(self, data: DatasetDict) -> DatasetDict:
        return DatasetDict(
            {
                "train": data["train"].shuffle().select(range(100)),
                "test": data["test"].shuffle().select(range(100)),
            }
        )

    def tokenise_fn(self, data: DatasetDict):
        return self.tokeniser(data["text"], padding="max_length", truncation=True)

    def postprocess(self, data: DatasetDict) -> DatasetDict:
        data = data.remove_columns(["text"])
        data = data.rename_column("label", "labels")
        data.set_format("torch")
        return data
