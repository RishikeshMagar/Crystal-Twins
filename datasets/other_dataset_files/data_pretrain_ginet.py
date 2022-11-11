import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing

import ase
# from ase.io import cif
from ase.io import read as ase_read
from pymatgen.core.structure import Structure

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_geometric.data import Data, Dataset, DataLoader
from torch_cluster import knn_graph

from datasets.atom_feat import AtomCustomJSONInitializer
from datasets.augmentation import RotationTransformation, \
    PerturbStructureTransformation, SwapAxesTransformation, \
    TranslateSitesTransformation


ATOM_LIST = list(range(1,100))
print("Number of atoms:", len(ATOM_LIST))

def get_all_cifs(data_dir):
    cif_fns = []
    for subdir, dirs, files in os.walk(data_dir):
        for fn in files:
            if fn.endswith('.cif'):
                cif_fns.append(os.path.join(subdir, fn))
    return cif_fns

class CrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=12, task='regression', fold = 0,atom_mask=0.25, edge_mask=0.0):
        self.k = k
        self.task = task
        self.root_dir = data_dir
        random_seed = 42
        #self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(self.root_dir), 'root_dir does not exist!'
        self.atom_mask = atom_mask
        self.edge_mask = edge_mask
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        self.cryst_files = get_all_cifs(data_dir)
        random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        random.shuffle(self.cryst_files)
        # atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        atom_init_file = os.path.join('datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.atom_featurizer = AtomCustomJSONInitializer(atom_init_file)
        # self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.rotater = RotationTransformation()
        self.perturber = PerturbStructureTransformation(distance=1, min_distance=0.0) 
        self.translator = TranslateSitesTransformation(None,None, True)      
        self.feat_dim = self.atom_featurizer.get_length() 
        #self.masker = RemoveSitesTransformation()

    def __getitem__(self, index):
        # get the cif id and path
        # read the cif file
        cryst_fn = self.cryst_files[index]
        # crys = ase_read(cryst_fn)
        crystal = Structure.from_file(cryst_fn)

        crystal_per_1 =  self.perturber.apply_transformation(crystal)
        crystal_per_2 =  self.perturber.apply_transformation(crystal)
        N = crystal.num_sites

        num_mask_nodes = max([1, math.floor(self.atom_mask*N)])
        mask_indices_i = random.sample(list(range(N)), num_mask_nodes)
        mask_indices_j = random.sample(list(range(N)), num_mask_nodes)

        crystal_rot_1 = crystal_per_1
        crystal_rot_2 = crystal_per_2




        pos_i = crystal_rot_1.frac_coords
        atom_indices_i = list(crystal_rot_1.atomic_numbers)
        cell_i = crystal_rot_1.lattice.get_cartesian_coords(1)
        feat_i = self.atom_featurizer.get_atom_features(atom_indices_i)
        N_i = len(pos_i)
        atomics_i = []
        for idx in atom_indices_i:
            atomics_i.append(ATOM_LIST.index(idx))
        atomics_i = torch.tensor(atomics_i, dtype=torch.long)
        pos_i = torch.tensor(pos_i, dtype=torch.float)
        feat_i = torch.tensor(feat_i, dtype=torch.float)
        edge_index_i = knn_graph(pos_i, k=self.k, loop=False)
        edge_attr_i = torch.zeros(edge_index_i.size(1), dtype=torch.long)

        pos_j = crystal_rot_2.frac_coords
        atom_indices_j = list(crystal_rot_2.atomic_numbers)
        cell_j = crystal_rot_2.lattice.get_cartesian_coords(1)
        feat_j = self.atom_featurizer.get_atom_features(atom_indices_j)
        N_j = len(pos_j)
        atomics_j = []
        for idx in atom_indices_j:
            atomics_j.append(ATOM_LIST.index(idx))
        atomics_j = torch.tensor(atomics_j, dtype=torch.long)
        pos_j = torch.tensor(pos_j, dtype=torch.float)
        feat_j = torch.tensor(feat_j, dtype=torch.float)
        edge_index_j = knn_graph(pos_j, k=self.k, loop=False)
        edge_attr_j = torch.zeros(edge_index_j.size(1), dtype=torch.long)






        # atomics_i, atomics_j = deepcopy(atomics_i), deepcopy(atomics_j)
        # pos_i, pos_j = deepcopy(pos_i), deepcopy(pos_i)
        
        for idx in mask_indices_i:
            atomics_i[idx] = torch.tensor(len(ATOM_LIST), dtype=torch.long)
            pos_i[idx,:] =  torch.zeros(3, dtype=torch.float)
            feat_i[idx,:] = torch.zeros(self.feat_dim, dtype=torch.float)
        
        for idx in mask_indices_j:
            atomics_j[idx] = torch.tensor(len(ATOM_LIST), dtype=torch.long)
            pos_j[idx,:] =  torch.zeros(3, dtype=torch.float)
            feat_i[idx,:] = torch.zeros(self.feat_dim, dtype=torch.float)


        # build the PyG graph 

        M_i = edge_index_i.size(1) // 2
        num_mask_edges_i = math.floor(self.edge_mask*M_i)

        mask_edges_single_i = random.sample(list(range(M_i)), num_mask_edges_i)
        mask_edges_i = [2*i for i in mask_edges_single_i] + [2*i+1 for i in mask_edges_single_i]
        masked_edge_index_i = torch.zeros((2, 2*(M_i-num_mask_edges_i)), dtype=torch.long)
        masked_edge_attr_i = torch.zeros(2*(M_i-num_mask_edges_i), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M_i):
            if bond_idx not in mask_edges_i:
                masked_edge_index_i[:,count] = edge_index_i[:,bond_idx]
                masked_edge_attr_i[count] = edge_attr_i[bond_idx]
                count += 1
        
        M_j = edge_index_j.size(1) // 2
        num_mask_edges_j = math.floor(self.edge_mask*M_j)


        mask_edges_single_j = random.sample(list(range(M_j)), num_mask_edges_j)
        mask_edges_j = [2*i for i in mask_edges_single_j] + [2*i+1 for i in mask_edges_single_j]
        masked_edge_index_j = torch.zeros((2, 2*(M_j-num_mask_edges_j)), dtype=torch.long)
        masked_edge_attr_j = torch.zeros(2*(M_j-num_mask_edges_j), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M_j):
            if bond_idx not in mask_edges_j:
                masked_edge_index_j[:,count] = edge_index_j[:,bond_idx]
                masked_edge_attr_j[count] = edge_attr_j[bond_idx]
                count += 1

        data_i = Data(
            atomics=atomics_i, pos=pos_i, feat=feat_i, 
            edge_index=masked_edge_index_i, edge_attr=masked_edge_attr_i
        )

        data_j = Data(
            atomics=atomics_j, pos=pos_j, feat=feat_j, 
            edge_index=masked_edge_index_j, edge_attr=masked_edge_attr_j
        )
        return data_i, data_j

    def __len__(self):
        return len(self.cryst_files)

class CrystalDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_dir='data', k=12, atom_mask = 0.25, edge_mask =0.25):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.atom_mask = atom_mask
        self.edge_mask = edge_mask

    def get_data_loaders(self):
        train_dataset = CrystalDataset(self.data_dir, self.k, self.atom_mask, self.edge_mask)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        #print(split)
        train_idx, valid_idx = indices[split:], indices[:split]
        #print("L",len(valid_idx))
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        #print(len(valid_loader))
        #print(len(train_loader))
        return train_loader, valid_loader


if __name__ == "__main__":
    dataset = CrystalDataset()
    print(dataset)
    print(dataset.__getitem__(0))
    dataset = CrystalDatasetWrapper(batch_size=2, num_workers=0, valid_size=0.1, test_size=0.1, data_dir='data/BG_cifs')
    train_loader, valid_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        print(data.atomics)
        print(data.pos)
        print(data.y)
        break