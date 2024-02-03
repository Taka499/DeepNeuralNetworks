import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'ctcdecode'))

from abc import ABC, abstractmethod
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as ttf
from tools import StoredModel, ModelSaver
from tools import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb

class Trainer(ABC):
    def __init__(self, 
                    model_id: str, 
                    batch_size: int, 
                    device, 
                    checkpoints: str, 
                    max_epochs: int = 10, 
                    epoch_start: int = 0, 
                    log_checkpoints:str = None, 
                    model_saver_mode = 'min', 
                    ) -> None:

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.max_epochs = max_epochs
        self.epoch_start = epoch_start
        self.epochs = epoch_start
        self.checkpoints = checkpoints
        if log_checkpoints:
            self.log_checkpoints = log_checkpoints
        else:
            self.log_checkpoints = checkpoints
        
        if not os.path.exists(self.checkpoints):
            os.mkdir(self.checkpoints)
        if not os.path.exists(self.log_checkpoints):
            os.mkdir(self.log_checkpoints)
        
        self.create_model_saver(checkpoints, mode=model_saver_mode)
        
        pass
    
    @abstractmethod
    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler):
        pass
    
    @abstractmethod
    def _train(self, model: nn.Module, train_loader: DataLoader, batch_bar: tqdm, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler):
        pass
    
    @abstractmethod
    def _val(self, model: nn.Module, val_loader: DataLoader, batch_bar: tqdm):
        pass
    
    @abstractmethod
    def inference(self):
        pass
    
    def create_model_saver(self, checkpoints, mode: str = 'min'):
        self.model_saver = ModelSaver(mode=mode, model_id=self.model_id, checkpoints=checkpoints)
    
    def save_spec(self, spec_str: str):
        self.model_saver.save_spec(spec_str)
    
    def generate_log(self, stats: dict, epoch_type: str = "train", epoch_time = None):
        if epoch_type == "train":
            log = f"Train Epoch {self.epochs}/{self.epoch_start+self.max_epochs}: "
        elif epoch_type == "val" or epoch_type == "validation":
            log = f" - Val Epoch {self.epochs}/{self.epoch_start+self.max_epochs}: "
        else:
            log = f" - {epoch_type} {self.epochs}/{self.epoch_start+self.max_epochs}: "

        for key in stats.keys():
            log = log + f"{str(key)}={stats[key]} | "

        if epoch_time:
            log = log + f"({int(epoch_time//60)}min{int(epoch_time%60)}s)"
        
        return log

    
    def save_log(self, log: str):
        with open(f"{self.log_checkpoints}/{self.model_id}/log.txt", mode='a') as f:
            f.write(f"{log}\n")
    
    def load_best_model(self):
        torch.cuda.empty_cache()
        _, model, _, _, _ = load_model(model_id=self.model_id, checkpoints=self.checkpoints, device=self.device, specific_epoch=self.model_saver.best_epoch)
        return model




class HW2P2Trainer(Trainer):
    def __init__(   self, 
                    model_id: str, 
                    batch_size: int, 
                    device, 
                    checkpoints: str, 
                    max_epochs: int = 10, 
                    epoch_start: int = 0, 
                    log_checkpoints:str = None, 
                    model_saver_mode = 'max', 
                    ) -> None:
        super().__init__(model_id, batch_size, device, checkpoints, max_epochs, epoch_start, log_checkpoints, model_saver_mode)

        self.train_losses = []
        self.verification_acc = []
    
    def _train(self, model: nn.Module, train_loader: DataLoader, batch_bar: tqdm, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        
        num_correct = 0

        for i, (images, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
            running_loss += float(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # scale = self.scaler.get_scale()
            scaler.update()
            scheduler.step()

            # skip_lr_sched = (scale > self.scaler.get_scale())
            
            # if not skip_lr_sched:
            #     

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / (self.batch_size*(i + 1))),
                loss = "{:.04f}".format(float(running_loss / (i+1))),
                num_correct=num_correct,
                lr = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            )
            batch_bar.update()
            
            del images
            del labels
            del outputs
            del loss
            torch.cuda.empty_cache()
        batch_bar.close()

        epoch_time = time.time() - start_time
        acc = 100 * num_correct / (self.batch_size* len(train_loader))
        train_loss = float(running_loss / len(train_loader))
        self.epochs += 1
        stats = {
            "acc": acc, 
            "loss": train_loss, 
            "lr": float(optimizer.param_groups[0]['lr']),
        }
        train_log = self.generate_log(stats=stats, epoch_type="train", epoch_time=epoch_time)

        self.train_losses.append(train_loss)
        self.save_log(train_log)

        return acc, train_loss


    
    def _val(self, model: nn.Module, val_loader: DataLoader, batch_bar: tqdm):
        model.eval()
        start_time = time.time()
        num_correct = 0


        for i, (images, labels) in enumerate(val_loader):
            torch.cuda.empty_cache()
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                outputs = model(images)
            
            num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())

            batch_bar.set_postfix(
                acc = "{:.04f}%".format(100 * num_correct / (self.batch_size*(i + 1))),
                num_correct = num_correct
            )
            batch_bar.update()
            del images
            del labels
            del outputs
            torch.cuda.empty_cache()
        batch_bar.close()
        
        epoch_time = time.time() - start_time
        acc = 100 * num_correct / (self.batch_size* len(val_loader))
        stats = {
            "acc": acc, 
        }
        val_log = self.generate_log(stats=stats, epoch_type="val", epoch_time=epoch_time)

        self.save_log(val_log)

        return acc
    
    def eval_verification(self, model, known_paths, unknown_images, known_images, similarity, batch_size=100, mode='val'):
        unknown_feats, known_feats = [], []

        batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
        model.eval()
        start_time = time.time()

        for i in range(0, unknown_images.shape[0], batch_size):
            unknown_batch = unknown_images[i:i+batch_size] # Slice a given portion upto batch_size
            
            with torch.no_grad():
                unknown_feat, _ = model(unknown_batch.float().to(self.device), return_feats=True) #Get features from model         
            unknown_feats.append(unknown_feat)
            batch_bar.update()
        
        batch_bar.close()

        batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)

        for i in range(0, known_images.shape[0], batch_size):
            known_batch = known_images[i:i+batch_size] 
            with torch.no_grad():
                known_feat, _ = model(known_batch.float().to(self.device), return_feats=True)
            
            known_feats.append(known_feat)
            batch_bar.update()

        batch_bar.close()

        # Concatenate all the batches
        unknown_feats = torch.cat(unknown_feats, dim=0)
        known_feats = torch.cat(known_feats, dim=0)

        similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
        # Print the inner list comprehension in a separate cell - what is really happening?

        predictions = similarity_values.argmax(0).cpu().numpy() #Why are we doing an argmax here?

        # Map argmax indices to identity strings
        pred_id_strings = [known_paths[i] for i in predictions]
        epoch_time = time.time() - start_time
        
        if mode == 'val':
            true_ids = pd.read_csv('/content/data/verification/dev_identities.csv')['label'].tolist()
            accuracy = accuracy_score(pred_id_strings, true_ids)
            self.verification_acc.append(accuracy)
            stats = {
                "acc": accuracy, 
            }
            val_log = self.generate_log(stats=stats, epoch_type="Verification", epoch_time=epoch_time)
            self.save_log(val_log)
            #print("Verification Accuracy = {}".format(accuracy))
            return accuracy
        
        return pred_id_strings

    
    def fit(self, model: nn.Module, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            optimizer: optim.Optimizer, 
            criterion: nn.Module, 
            scheduler: optim.lr_scheduler._LRScheduler, 
            scaler: torch.cuda.amp.GradScaler, 
            known_paths, 
            unknown_images, 
            known_images, 
            similarity_metric, 
            wandb, 
            wandb_run):
        total_epochs = self.epoch_start + self.max_epochs

        for epoch in range(self.epoch_start, total_epochs):

            curr_lr = float(optimizer.param_groups[0]['lr'])

            batch_bar_train = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train Epoch {epoch+1}/{total_epochs}:")
            train_acc, train_loss = self._train(model, train_loader, batch_bar_train, optimizer, criterion, scheduler, scaler)

            batch_bar_val = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Val Epoch {epoch+1}:")
            val_acc = self._val(model, val_loader, batch_bar_val)

            # verification
            verification_dev_acc = self.eval_verification(model, known_paths, unknown_images, known_images, similarity_metric, 100, mode='val')

            # evaluation

            wandb.log({"train_loss":train_loss, 'train_Acc': train_acc, 'validation_Acc':val_acc, 
               "verification_acc": verification_dev_acc, "learning_Rate": curr_lr})

            stats = {
                "epoch": epoch+1,
                "lr": float(optimizer.param_groups[0]['lr']),
                "train_acc": train_acc, 
                "train_loss": train_loss,
                "val_acc": val_acc, 
                "verification_acc": verification_dev_acc, 
            }
            self.model_saver.save(StoredModel(model, optimizer, scheduler, criterion), stats, val_acc)

        print(f"The best epoch is: {self.model_saver.best_epoch}")
        with open(f"{self.log_checkpoints}/{self.model_id}/log.txt", mode='a') as f:
            f.write(f"The best epoch is: {self.model_saver.best_epoch}\n")
        
        wandb_run.finish()

        # plot the loss curve
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.train_losses, label = "train loss")
        # plt.plot(range(self.epoch_start+1, total_epochs+1), self.val_losses, label = "val loss")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("Loss")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/loss_curves_{self.model_id}.png")
        plt.show()

        # plot the verification_acc curve
        #plt.plot(range(1, epochs+1), train_lev, label = "train lev_dist")
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.verification_acc, label = "verification_acc")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("verification_acc")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/verification_acc_curves_{self.model_id}.png")
        plt.show()
    
    def inference(self, test_loader, test_data):
        pass


class HW2P2TrainerWithFineTuneLoss(Trainer):
    def __init__(   self, 
                    model_id: str, 
                    batch_size: int, 
                    device, 
                    checkpoints: str, 
                    max_epochs: int = 10, 
                    epoch_start: int = 0, 
                    log_checkpoints:str = None, 
                    model_saver_mode = 'max', 
                    ) -> None:
        super().__init__(model_id, batch_size, device, checkpoints, max_epochs, epoch_start, log_checkpoints, model_saver_mode)

        self.train_losses0 = []
        self.train_losses1 = []
        self.verification_acc = []
    
    def _train(self, 
               model: nn.Module, 
               train_loader: DataLoader, 
               batch_bar: tqdm, 
               optimizer: optim.Optimizer, 
               criterion: nn.Module, 
               scheduler: optim.lr_scheduler._LRScheduler, 
               scaler: torch.cuda.amp.GradScaler, 
               fine_tuning_loss: nn.Module, 
               optimizer_loss: optim.Optimizer, 
               loss_weight: float, 
               ):
        model.train()
        start_time = time.time()
        running_loss0 = 0.0
        running_loss1 = 0.0
        
        num_correct = 0

        for i, (images, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            optimizer_loss.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast():
                feats, outputs = model(images, return_feats=True)
                # loss0 = criterion(outputs, labels)
                loss1 = loss_weight * fine_tuning_loss(feats, labels)
            
            num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
            # running_loss0 += float(loss0.item())
            running_loss1 += float(loss1.item())

            # scaler.scale(loss0).backward(retain_graph=True)
            scaler.scale(loss1).backward()
            # update fine tuning loss' parameters
            for parameter in fine_tuning_loss.parameters():
                parameter.grad.data *= (1.0 / loss_weight)

            scaler.step(optimizer_loss)
            scaler.step(optimizer)
            # scale = self.scaler.get_scale()
            scaler.update()
            
            scheduler.step()

            # skip_lr_sched = (scale > self.scaler.get_scale())
            
            # if not skip_lr_sched:
            #     

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / (self.batch_size*(i + 1))),
                # Xent_loss = "{:.04f}".format(float(running_loss0 / (i+1))),
                fine_tune_loss = "{:.04f}".format(float(running_loss1 / (i+1))),
                num_correct = num_correct,
                lr = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            )
            batch_bar.update()
            
            del images
            del labels
            del outputs
            # del loss0
            del loss1
            torch.cuda.empty_cache()
        batch_bar.close()

        epoch_time = time.time() - start_time
        acc = 100 * num_correct / (self.batch_size* len(train_loader))
        train_Xent_loss = float(running_loss0 / len(train_loader))
        train_fine_tune_loss = float(running_loss1 / len(train_loader))
        self.epochs += 1
        stats = {
            "acc": acc, 
            "Xent_loss": train_Xent_loss, 
            "fine_tune_loss": train_fine_tune_loss, 
            "lr": float(optimizer.param_groups[0]['lr']),
        }
        train_log = self.generate_log(stats=stats, epoch_type="train", epoch_time=epoch_time)

        # self.train_losses0.append(train_Xent_loss)
        self.train_losses1.append(train_fine_tune_loss)
        self.save_log(train_log)

        return acc, train_Xent_loss, train_fine_tune_loss


    
    def _val(self, model: nn.Module, val_loader: DataLoader, batch_bar: tqdm):
        model.eval()
        start_time = time.time()
        num_correct = 0


        for i, (images, labels) in enumerate(val_loader):
            torch.cuda.empty_cache()
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                outputs = model(images)
            
            num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())

            batch_bar.set_postfix(
                acc = "{:.04f}%".format(100 * num_correct / (self.batch_size*(i + 1))),
                num_correct = num_correct
            )
            batch_bar.update()
            del images
            del labels
            del outputs
            torch.cuda.empty_cache()
        batch_bar.close()
        
        epoch_time = time.time() - start_time
        acc = 100 * num_correct / (self.batch_size* len(val_loader))
        stats = {
            "acc": acc, 
        }
        val_log = self.generate_log(stats=stats, epoch_type="val", epoch_time=epoch_time)

        self.save_log(val_log)

        return acc
    
    def eval_verification(self, model, known_paths, unknown_images, known_images, similarity, batch_size=100, mode='val'):
        unknown_feats, known_feats = [], []

        batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
        model.eval()
        start_time = time.time()

        for i in range(0, unknown_images.shape[0], batch_size):
            unknown_batch = unknown_images[i:i+batch_size] # Slice a given portion upto batch_size
            
            with torch.no_grad():
                unknown_feat, _ = model(unknown_batch.float().to(self.device), return_feats=True) #Get features from model         
            unknown_feats.append(unknown_feat)
            batch_bar.update()
        
        batch_bar.close()

        batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)

        for i in range(0, known_images.shape[0], batch_size):
            known_batch = known_images[i:i+batch_size] 
            with torch.no_grad():
                known_feat, _ = model(known_batch.float().to(self.device), return_feats=True)
            
            known_feats.append(known_feat)
            batch_bar.update()

        batch_bar.close()

        # Concatenate all the batches
        unknown_feats = torch.cat(unknown_feats, dim=0)
        known_feats = torch.cat(known_feats, dim=0)

        similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
        # Print the inner list comprehension in a separate cell - what is really happening?

        predictions = similarity_values.argmax(0).cpu().numpy() #Why are we doing an argmax here?

        # Map argmax indices to identity strings
        pred_id_strings = [known_paths[i] for i in predictions]
        epoch_time = time.time() - start_time
        
        if mode == 'val':
            true_ids = pd.read_csv('/content/data/verification/dev_identities.csv')['label'].tolist()
            accuracy = accuracy_score(pred_id_strings, true_ids)
            self.verification_acc.append(accuracy)
            stats = {
                "acc": accuracy, 
            }
            val_log = self.generate_log(stats=stats, epoch_type="Verification", epoch_time=epoch_time)
            self.save_log(val_log)
            #print("Verification Accuracy = {}".format(accuracy))
            return accuracy
        
        return pred_id_strings

    
    def fit(self, 
            model: nn.Module, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            optimizer: optim.Optimizer, 
            criterion: nn.Module, 
            scheduler: optim.lr_scheduler._LRScheduler, 
            scaler: torch.cuda.amp.GradScaler, 
            fine_tuning_loss: nn.Module, 
            optimizer_loss: optim.Optimizer, 
            loss_weight: float, 
            known_paths, 
            unknown_images, 
            known_images, 
            similarity_metric, 
            wandb, 
            wandb_run):
        total_epochs = self.epoch_start + self.max_epochs

        for epoch in range(self.epoch_start, total_epochs):

            curr_lr = float(optimizer.param_groups[0]['lr'])

            batch_bar_train = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train Epoch {epoch+1}/{total_epochs}:")
            train_acc, train_Xent_loss, train_fine_tune_loss = self._train(model, train_loader, batch_bar_train, optimizer, criterion, scheduler, scaler, fine_tuning_loss, optimizer_loss, loss_weight)

            batch_bar_val = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Val Epoch {epoch+1}:")
            val_acc = self._val(model, val_loader, batch_bar_val)

            # verification
            verification_dev_acc = self.eval_verification(model, known_paths, unknown_images, known_images, similarity_metric, 100, mode='val')

            # evaluation

            wandb.log({"train_Xent_loss": train_Xent_loss, "train_fine_tune_loss": train_fine_tune_loss, 'train_Acc': train_acc, 'validation_Acc':val_acc, 
               "verification_acc": verification_dev_acc, "learning_Rate": curr_lr})

            stats = {
                "epoch": epoch+1,
                "lr": float(optimizer.param_groups[0]['lr']),
                "train_acc": train_acc, 
                "train_Xent_loss": train_Xent_loss, 
                "train_fine_tune_loss": train_fine_tune_loss,
                "val_acc": val_acc, 
                "verification_acc": verification_dev_acc, 
            }
            self.model_saver.save(StoredModel(model, optimizer, scheduler, criterion), stats, val_acc)

        print(f"The best epoch is: {self.model_saver.best_epoch}")
        with open(f"{self.log_checkpoints}/{self.model_id}/log.txt", mode='a') as f:
            f.write(f"The best epoch is: {self.model_saver.best_epoch}\n")
        
        wandb_run.finish()

        # plot the loss curve
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.train_losses0, label = "train loss")
        # plt.plot(range(self.epoch_start+1, total_epochs+1), self.val_losses, label = "val loss")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("Xent Loss")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/Xent_loss_curves_{self.model_id}.png")
        plt.show()
        
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.train_losses1, label = "train loss")
        # plt.plot(range(self.epoch_start+1, total_epochs+1), self.val_losses, label = "val loss")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("Fine Tune Loss")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/fine_tune_loss_curves_{self.model_id}.png")
        plt.show()

        # plot the verification_acc curve
        #plt.plot(range(1, epochs+1), train_lev, label = "train lev_dist")
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.verification_acc, label = "verification_acc")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("verification_acc")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/verification_acc_curves_{self.model_id}.png")
        plt.show()
    
    def inference(self, test_loader, test_data):
        pass

