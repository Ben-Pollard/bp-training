import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler

from bp_training.data import YelpReviewBertData

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )
    model.to(device)

    data_loader = YelpReviewBertData().train_dataloader

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):

        for batch in data_loader:

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            lr_scheduler.step()

            optimizer.zero_grad()

            progress_bar.update(1)
