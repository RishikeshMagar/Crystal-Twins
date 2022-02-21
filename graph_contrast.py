import os
import shutil
import sys
import torch
import numpy as np
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.gin import GINet
from loss.barlow_twins import BarlowTwinsLoss


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_contrast.yaml', os.path.join(model_checkpoints_folder, 'config_contrast.yaml'))


class GraphContrastive(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        # self.writer = SummaryWriter()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_contrast', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        self.criterion = BarlowTwinsLoss(self.device, config['batch_size'], **config['loss'])


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data_i, data_j):
        # get the representations and the projections
        _, zis = model(data_i)  # [N,C]
        # get the representations and the projections
        _, zjs = model(data_j)  # [N,C]

        # normalize projection feature vectors
        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = GINet(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['init_lr'], weight_decay=eval(self.config['weight_decay']))
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (data_i, data_j) in enumerate(train_loader):
                optimizer.zero_grad()

                data_i, data_j = data_i.to(self.device), data_j.to(self.device)
                loss = self._step(model, data_i, data_j)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()

                optimizer.step()
                n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    scheduler.step()

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print('Validation', valid_loss)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter > 0 and epoch_counter % 2 == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
            
            # restart CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        #print(len(valid_loader))
        with torch.no_grad():
            model.eval()

            loss_total = 0.0
            counter = 0
            for data_i, data_j in valid_loader:
                #print(data_i)
                #print(data_j)
                data_i, data_j = data_i.to(self.device), data_j.to(self.device)
                loss = self._step(model, data_i, data_j)
                loss_total += loss.item()
                counter += 1

            loss_total /= counter
        
        model.train()
        return loss_total

