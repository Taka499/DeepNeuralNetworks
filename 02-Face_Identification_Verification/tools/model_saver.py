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
    def __init__(self, mode, model_id: str, checkpoints: str, n_latest=3):
        self.mode = mode
        self.checkpoints = checkpoints
        self.model_id = model_id
        self.n_latest = n_latest
        self.best_epoch = 0
        self.latest_epoch = 0
        if mode == "min":
            self.best_metric = float("inf")
            self.is_better = lambda x: x < self.best_metric
        elif mode == "max":
            self.best_metric = float("-inf")
            self.is_better = lambda x: x > self.best_metric
        else:
            raise Exception(f"Unsupported mode: {mode}")

        if not os.path.exists(self.checkpoints):
            os.mkdir(self.checkpoints)
        if not os.path.exists(f"{checkpoints}/{self.model_id}"):
            os.mkdir(f"{checkpoints}/{self.model_id}")
        else:
            raise Exception(f"{self.model_id} already exists. Please delete the checkpoints or change the model id.")
        self.model_path = f"{self.checkpoints}/{self.model_id}"
    
    def save_spec(self, input: str):
        if not os.path.exists(f"{self.checkpoints}/{self.model_id}/model_spec.txt"):
            with open(f"{self.checkpoints}/{self.model_id}/model_spec.txt", mode="w") as f:
                f.write(input + "\n")
        else:
            with open(f"{self.checkpoints}/{self.model_id}/model_spec.txt", mode="a") as f:
                f.write(input + "\n")
    
    def save(self, stored_model, epoch_stats_dict, metric):
        """
            Save the parameters of the 'n' latest and the best epochs
        """
        epoch = epoch_stats_dict["epoch"]
        path = f"{self.model_path}/epoch_{epoch}.model"
        
        self.latest_epoch = epoch
        torch.save(stored_model, path)

        # write the log of stats
        with open(f"{self.model_path}/stats_log.txt", mode='a') as log_file:
            log_file.write(json.dumps(epoch_stats_dict) + "\n")
        
        if self.is_better(metric):
            self.best_metric = metric
            self.best_epoch = epoch

        self.delete_prev_checkpoints()

    def delete_prev_checkpoints(self):
        for filename in os.listdir(self.model_path):
            if '.model' in filename:
                epoch = int(filename[6:-6])
                if epoch == self.best_epoch:
                    continue
                elif epoch > self.latest_epoch-self.n_latest:
                    continue
                else:
                    os.remove(f"{self.model_path}/{filename}")
