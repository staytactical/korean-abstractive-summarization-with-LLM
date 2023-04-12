import torch
from transformers import get_linear_schedule_with_warmup

def train_model(peft_model, train_loader, num_epochs, learning_rate):
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    peft_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        tr_loss = torch.tensor(0.0).to('cuda')
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = peft_model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_idx % 500 == 0:
                train_epoch_loss = total_loss / batch_idx
                print(f"{epoch=}: {train_epoch_loss=}")

        peft_model.save_pretrained(f'./GPT_Competition/T5_Lora_aug_{epoch}epoch')
