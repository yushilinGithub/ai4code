import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from config import Config
from sklearn.metrics import mean_squared_error,f1_score
import os
import numpy as np
class Trainer:
    def __init__(self, config, dataloaders, optimizer, model, loss_fns, scheduler, device="cuda:0"):
        self.train_loader, self.valid_loader = dataloaders
        self.train_loss_fn, self.valid_loss_fn = loss_fns
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.device = torch.device(device)
        self.config = config

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        accs = []

        for bnum, cache in train_pbar:
            ids = self._convert_if_not_tensor(cache[0], dtype=torch.long)
            mask = self._convert_if_not_tensor(cache[1], dtype=torch.long)
            ids_code = self._convert_if_not_tensor(cache[2], dtype=torch.long)

            
            with autocast(enabled=True):
                #outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)
                outputs_text,outputs_code = self.model(ids=ids, mask=mask, ids_code=ids_code)
                loss = self.model.loss(outputs_text,outputs_code)

                train_pbar.set_description('loss: {:.2f},acc: {:.2f}'.format(loss["loss"],loss["acc"]))

                Config.scaler.scale(loss["loss"]).backward()
                Config.scaler.step(self.optimizer)
                Config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                            
            accs.append(loss["acc"])
        
        # Tidy

        torch.cuda.empty_cache()
        
        return accs

    @torch.no_grad()
    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        valid_pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        accs = []


        for bnum, cache in valid_pbar:
            ids = self._convert_if_not_tensor(cache[0], dtype=torch.long)
            mask = self._convert_if_not_tensor(cache[1], dtype=torch.long)
            ids_code = self._convert_if_not_tensor(cache[2], dtype=torch.long)

            
            with autocast(enabled=True):
                #outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)
                outputs_text,outputs_code = self.model(ids=ids, mask=mask, ids_code=ids_code)
                loss = self.model.loss(outputs_text,outputs_code)

                valid_pbar.set_description('loss: {:.2f},acc: {:.2f}'.format(loss["loss"],loss["acc"]))

 
            accs.append(loss["acc"])
        
        # Tidy

        torch.cuda.empty_cache()
        
        return accs


    def fit(self, epochs: int = 10, output_dir: str = "/kaggle/working/", custom_name: str = 'model.pth'):
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_f1 = int(1e+7)
        best_preds = None
        for epx in range(epochs):
            print(f"{'='*20} Epoch: {epx+1} / {epochs} {'='*20}")

            accs = self.train_one_epoch()
            train_accs = np.mean(accs)
            #train_mse = mean_squared_error(train_targets, train_preds)
            print(f"Training loss: {train_accs:.4f}")

            val_accs = self.valid_one_epoch()
            #valid_mse = mean_squared_error(valid_targets, valid_preds)
            val_accs = np.mean(val_accs)
            print(f"Validation loss: {val_accs:.4f}")
            

            
            if val_accs > best_f1:
                best_f1 = val_accs
                self.save_model(output_dir, custom_name)
                print(f"Saved model with val_loss: {best_f1:.4f}")
            
    def save_model(self, path, name, verbose=False):
        """
        Saves the model at the provided destination
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        torch.save(self.model.state_dict(), os.path.join(path, name))
        if verbose:
            print(f"Model Saved at: {os.path.join(path, name)}")

    def _convert_if_not_tensor(self, x, dtype):
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)
