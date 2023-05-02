import os
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets.data_pretrain_mask import CIFData
from datasets.data_pretrain_mask import collate_pool, get_train_val_test_loader
from models.cgcnn_pretrain_sim import CrystalGraphConvNet


import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class CrystalContrastive(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_contrast', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
    

        self.dataset = CIFData(**self.config['dataset'])
        collate_fn = collate_pool
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            pin_memory=self.config['cuda'],
            batch_size=self.config['batch_size'], 
            **self.config['dataloader']
        )

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print("GPU available")
        else:
            print("GPU not available")
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'#torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
            #torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def criterion(self,q, k):
        return  nn.CosineSimilarity(dim=1)(q,k).to(self.device)

    def _step(self, model, data_i, data_j):
        # get the representations and the projections
        zis,pis = model(*data_i)  # [N,C]
        # get the representations and the projections
        zjs,pjs = model(*data_j)  # [N,C]

        # normalize projection feature vectors
        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)

        return pis,pjs,zis.detach(),zjs.detach()

    def train(self):

        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    **self.config['model']
        )
        model = self._load_pre_trained_weights(model)
        if self.config['cuda'] and torch.cuda.device_count()>1:
            #model = nn.DataParallel(model, device_ids = [0,1,2])
            model = model.to(self.device)
        elif self.config['cuda']:
            model = model.to(self.device)
    
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), self.config['optim']['lr'],
                                momentum=self.config['optim']['momentum'],
                                weight_decay=eval(self.config['optim']['weight_decay']))
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), self.config['optim']['lr'],
                                weight_decay=eval(self.config['optim']['weight_decay']))
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (input_1, input_2, _) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_var_rot_1 = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                Variable(input_1[1].to(self.device, non_blocking=True)),
                                input_1[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])

                    input_var_rot_2 = (Variable(input_2[0].to(self.device, non_blocking=True)),
                                Variable(input_2[1].to(self.device, non_blocking=True)),
                                input_2[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_2[3]])
                else:
                    input_var_rot_1 = (Variable(input_1[0]),
                                Variable(input_1[1]),
                                input_1[2],
                                input_1[3])
                    input_var_rot_2 = (Variable(input_2[0]),
                                Variable(input_2[1]),
                                input_2[2],
                                input_2[3])
                
                p1, p2, z1, z2 = self._step(model, input_var_rot_1, input_var_rot_2)

                loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            torch.cuda.empty_cache()
            #print("1st",os.system('free -h'))  
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, self.valid_loader)
                print('Validation', valid_loss)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter > 0 and epoch_counter % 1 == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
            
            # warmup for the first 5 epochs
            if epoch_counter >= 5:
                scheduler.step()
    
    def _load_pre_trained_weights(self, model):
        #print("Here")
        try:
            checkpoints_folder = os.path.join('./runs_contrast', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_5.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
            #print("loaded")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        #print("2st",os.system('free -h'))
        with torch.no_grad():
            model.eval()

            loss_total = 0.0
            total_num = 0
            for input_1, input_2, batch_cif_ids in valid_loader:
                if self.config['cuda']:
                    input_var_rot_1 = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                Variable(input_1[1].to(self.device, non_blocking=True)),
                                input_1[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])

                    input_var_rot_2 = (Variable(input_2[0].to(self.device, non_blocking=True)),
                                Variable(input_2[1].to(self.device, non_blocking=True)),
                                input_2[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_2[3]])
                else:
                    input_var_rot_1 = (Variable(input_1[0]),
                                Variable(input_1[1]),
                                input_1[2],
                                input_1[3])
                    input_var_rot_2 = (Variable(input_2[0]),
                                Variable(input_2[1]),
                                input_2[2],
                                input_2[3])
                #print("3rd",os.system('free -h'))
                p1, p2, z1, z2 = self._step(model, input_var_rot_1, input_var_rot_2)
                loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
                loss_total += loss.item() * len(batch_cif_ids)
                total_num += len(batch_cif_ids)
        
            loss_total /= total_num
        #print("4th",os.system('free -h'))
        torch.cuda.empty_cache()
        model.train()
        return loss_total



if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    crys_contrast = CrystalContrastive(config)
    crys_contrast.train()
