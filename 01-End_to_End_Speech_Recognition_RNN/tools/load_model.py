import torch
import os

# Also cited from Zhe Chen's experience sharing
def load_model(model_id: str, checkpoints: str, device, specific_epoch=None):
    #checkpoints = "/content/drive/MyDrive/CMU/Spring2022/11785-Intro-to-Deep-Learning/hw2/model_checkpoints"
    epoch_start = -1
    for checkpoint in os.listdir(f"{checkpoints}/{model_id}"):
        if not checkpoint.startswith("epoch"):
            continue
        epoch = int(checkpoint.split("_")[1])
        if specific_epoch is None:
            if epoch > epoch_start:
                epoch_start = epoch
                last_checkpoint = checkpoint
        else:
            if epoch == specific_epoch:
                epoch_start = epoch
                last_checkpoint = checkpoint
                break

    if epoch_start == -1:
        print(f"No checkpoints available for {model_id}")
    else:
        epoch_start += 1
        print(f"resuming from last_checkpoint {last_checkpoint}")
        data = torch.load(f"{checkpoints}/{model_id}/{last_checkpoint}", map_location=device)
    model = data.model
    optimizer = data.optimizer
    scheduler = data.scheduler
    criterion = data.criterion

    return epoch_start, model, optimizer, scheduler, criterion