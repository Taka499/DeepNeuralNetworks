import torch
import json
import os

## Cite from Zhe Chen's experience sharing: piazza post @672
class StoredModel:
    def __init__(self, model, optimizer, scheduler, criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

class ModelSaver():
    def __init__(self, mode, model_id: str, checkpoints: str, regular_save_interval=5):
        self.mode = mode
        self.checkpoints = checkpoints
        self.model_id = model_id
        self.regular_save_interval = regular_save_interval
        self.prev_model_paths = list()
        self.best_epoch = 0
        if mode == "min":
            self.best_metric = float("inf")
            self.is_better = lambda x: x < self.best_metric
        elif mode == "max":
            self.best_metric = float("-inf")
            self.is_better = lambda x: x > self.best_metric
        else:
            raise Exception(f"Unsupported mode: {mode}")
    
    def save(self, stored_model, epoch_stats_dict, metric):
        epoch = epoch_stats_dict["epoch"]
        model_path = f"{self.checkpoints}/{self.model_id}"
        path = f"{model_path}/epoch_{epoch}"

        torch.save(stored_model, path)

        with open(f"{model_path}/training_logs.txt", mode='a') as log_file:
            log_file.write(json.dumps(epoch_stats_dict) + "\n")
        
        if self.is_better(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.delete_prev_checkpoints()
        else:
            if (epoch - self.best_epoch) % self.regular_save_interval == 0:
                self.prev_model_paths = self.prev_model_paths[1:]
                self.delete_prev_checkpoints()
        
        self.prev_model_paths.append(path)

    def delete_prev_checkpoints(self):
        for path in self.prev_model_paths:
            os.remove(path)
        self.prev_model_paths = list()
