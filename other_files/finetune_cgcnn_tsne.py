import os
import csv
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
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from dataset.data_tsne import CIFData
from dataset.data_tsne import collate_pool, get_data_loader
from model.cgcnn_finetune_tsne import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_tsne.yaml', os.path.join(model_checkpoints_folder, 'config_tsne.yaml'))


class FineTune(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_ft', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        if self.config['task'] == 'classification':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.MSELoss()

        self.dataset = CIFData(self.config['task'], **self.config['dataset'])
        collate_fn = collate_pool
        self.train_loader  = get_data_loader(
            dataset = self.dataset,
            collate_fn = collate_fn,
            pin_memory = self.config['cuda'],
            batch_size = self.config['batch_size'], 
            **self.config['dataloader']
        )

        # obtain target value normalizer
        if self.config['task'] == 'classification':
            self.normalizer = Normalizer(torch.zeros(2))
            self.normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            if len(self.dataset) < 500:
                warnings.warn('Dataset has less than 500 data points. '
                            'Lower accuracy is expected. ')
                sample_data_list = [self.dataset[i] for i in range(len(self.dataset))]
            else:
                sample_data_list = [self.dataset[i] for i in
                                    sample(range(len(self.dataset)), 500)]
            _, sample_target, _ = collate_pool(sample_data_list)
            self.normalizer = Normalizer(sample_target)

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def get_finetune_embedding(self):
        # train_loader, valid_loader = self.dataset.get_data_loaders()

        # model = GINet(**self.config["model"]).to(self.device)
        # model = self._load_pre_trained_weights(model)

        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    classification=(self.config['task']=='classification'), 
                                    **self.config['model']
        )
        if self.config['cuda']:
            model = model.to(self.device)

        model = self._load_pre_trained_weights(model)
        #print(len(model))
        #pytorch_total_params = sum(p.numel() for p in model.parameters if p.requires_grad)
        #print(pytorch_total_params)
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))      
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        output_embed = []
        cif_ids = [] 
        ground_truth = []

        for epoch_counter in range(1):
            for bn, (input, target, cif_id) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
                
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output,embed = model(*input_var)

                output = model(*input_var)
                cif_ids.append(cif_id)
                ground_truth.append(target) 
                output_embed.append(embed)

        output_tensor = torch.cat(output_embed)
        output_cif_id = np.concatenate(cif_ids)

        return output_tensor,output_cif_id, ground_truth
           
    def _load_pre_trained_weights(self, model):
        # try:
        #     checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
        #     state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        #     model.load_state_dict(state_dict)
        #     print("Loaded pre-trained model with success.")
        # except FileNotFoundError:
        #     print("Pre-trained weights not found. Training from scratch.")

        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
            load_state = torch.load(os.path.join(checkpoints_folder, 'model.pth'),  map_location=self.config['gpu']) 
 
            # checkpoint = torch.load('model_best.pth.tar', map_location=args.gpu)
            # load_state = checkpoint['state_dict']
            model_state = model.state_dict()

            #pytorch_total_params = sum(p.numel() for p in model_state.parameters if p.requires_grad)
            #print(pytorch_total_params)
            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded:', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

if __name__ == "__main__":
    config = yaml.load(open("config_tsne.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if 'band' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'band'
    elif 'fermi' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'fermi'
    elif 'Is_Metal_cifs' in config['dataset']['root_dir']:
        config['task'] = 'classification'
        task_name = 'Is_Metal_cifs'
    elif 'MP-formation-energy' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'MP_energy'
    elif 'lanths' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'lanths'
    elif 'CSD_Structures' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'csd'
    elif 'phonons' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'phonons'
    elif 'perovskites' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'perovskites'
    elif 'FE' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'FE'
    elif 'GVRH' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'GVRH'
    elif 'HOIP' in config['dataset']['root_dir']:
        config['task'] = 'regression'
        task_name = 'HOIP'

    fine_tune = FineTune(config)
    output_embed, cif_ids, ground_truth = fine_tune.get_finetune_embedding()
    output_embed_arr = output_embed.detach().cpu().numpy()
    ground_truth_arr = ground_truth.detach().cpu().numpy()
    #np.save("perovskites_embedding_ft.npy",output_embed_arr)
    #np.save("CIF_ID_Perovkites_ft.npy", cif_ids)
