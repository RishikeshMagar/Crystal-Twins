import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
import os, sys
import math
import re
import functools
from pymatgen.core.structure import Structure,SiteCollection
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.analysis.local_env import VoronoiNN,MinimumDistanceNN
from ase.io import read,write

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=32, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    if kwargs['train_size']:
        train_size = kwargs['train_size']
        print(train_size)
    else:
        train_size = int(train_ratio * total_size)
        print(train_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    # Generate random indices every time
    indices = np.arange(total_size)
    mask = np.ones(len(indices),dtype = bool)
    indices_train = np.random.choice(indices, train_size,replace=False)
    mask[indices_train] = False
    indices_remain = indices[mask]
    indices_valid = np.random.choice(indices_remain,valid_size,replace=False)
    ind_test = []
    for i in range(len(indices_remain)):
        if indices_remain[i] not in indices_valid:
            ind_test.append(indices_remain[i])
    
    indices_test = np.asarray(ind_test)
    
    train_sampler = SubsetRandomSampler(indices_train)
    val_sampler = SubsetRandomSampler(indices_valid)
    if return_test:
        test_sampler = SubsetRandomSampler(indices_test)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader



def collate_pool(dataset_list): # batch the crystal atoms
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
        ([atom_fea, hot_fea], nbr_fea, nbr_fea_idx, target)

        atom_fea: torch.Tensor shape(n_i, atom_fea_len)
        hot_fea: torch.Tensor shape(n_i, hot_fea_len)
        nbr_fea: torch.Tensor shape(n_i, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape(n_i, M)
        target: torch.Tensor shape(1,)
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_hot_fea: torch.Tensor shape (N, hot_fea_len)
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_hot_fea = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, (([atom_fea,hot_fea],nbr_fea,nbr_fea_idx),target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_hot_fea.append(hot_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return ([torch.cat(batch_atom_fea, dim=0),
             torch.cat(batch_hot_fea, dim=0)],
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

def calculateDistance(a,b):   # Atom-wise OFM
    dist =math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    return dist

def make_hot_for_atom_i(crystal,i,hvs):
    EP = str(crystal[i].specie)
    HV_P = np.nan_to_num(hvs[EP])
    AA = HV_P.reshape((HV_P.shape[1], 32))
    A = np.array(AA)
    b=VoronoiNN().get_nn_info(crystal,i)
    angles = []
    for nb in b:
        angle_K = nb['poly_info']['solid_angle']
        angles.append(angle_K)
    max_angle = max(angles)
    X_P = np.zeros(shape=(32,32))
    tmp_X = []
    for nb in b:
        EK = str(nb['site'].specie)
        angle_K = nb['poly_info']['solid_angle']
        index_K = nb['site_index']
        r_pk = ((calculateDistance(nb['site'].coords,crystal[i].coords))*(calculateDistance(nb['site'].coords,crystal[i].coords)))
        HV_K = hvs[EK]
        HV_K = HV_K.reshape((HV_K.shape[1], 32))
        coef_K = (angle_K/max_angle)*((1/((r_pk)**2)))
        HV_K_new= np.nan_to_num(coef_K * HV_K)
        X_PT = np.matmul(HV_P, HV_K_new)
        tmp_X.append(X_PT)
    X0 = np.zeros(shape=(32,32))
    for el in tmp_X:
        X0 = [[sum(x) for x in zip(el[i], X0[i])] for i in range(len(el))]
    X0  = np.concatenate((A.T,X0),axis = 1)
    X0 = np.asarray(X0)
    return X0 

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        ##print("Filter:",self.filter,len(self.filter))
        ##print("Var:",self.var)
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        ##print(atom_types)
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=2):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
   
        atom_fea = torch.Tensor(atom_fea)
        elements = {'H':['1s2'],'Li':['[He] 1s2'],'Be':['[He] 2s2'],'B':['[He] 2s2 2p1'],'N':['[He] 2s2 2p3'],'O':['[He] 2s2 2p4'],
                     'C':['[He] 2s2 2p2'], 'I':['[Kr] 4d10 5s2 5p5'],
                     'F':['[He] 2s2 2p5'],'Na':['[Ne] 3s1'],'Mg':['[Ne] 3s2'],'Al':['[Ne] 3s2 3p1'],'Si':['[Ne] 3s2 3p2'],
                     'P':['[Ne] 3s2 3p3'],'S':['[Ne] 3s2 3p4'],'Cl':['[Ne] 3s2 3p5'],'K':['[Ar] 4s1'],'Ca':['[Ar] 4s2'],'Sc':['[Ar] 3d1 4s2'],
                     'Ti':['[Ar] 3d2 4s2'],'V':['[Ar] 3d3 4s2'],'Cr':['[Ar] 3d5 4s1'],'Mn':['[Ar] 3d5 4s2'],
                     'Fe':['[Ar] 3d6 4s2'],'Co':['[Ar] 3d7 4s2'],'Ni':['[Ar] 3d8 4s2'],'Cu':['[Ar] 3d10 4s1'],'Zn':['[Ar] 3d10 4s2'],
                     'Ga':['[Ar] 3d10 4s2 4p2'],'Ge':['[Ar] 3d10 4s2 4p2'],'As':['[Ar] 3d10 4s2 4p3'],'Se':['[Ar] 3d10 4s2 4p4'],'Br':['[Ar] 3d10 4s2 4p5'],'Rb':['[Kr] 5s1'],
                     'Sr':['[Kr] 5s2'],'Y':['[Kr] 4d1 5s2'],'Zr':['[Kr] 4d2 5s2'],'Nb':['[Kr] 4d4 5s1'],'Mo':['[Kr] 4d5 5s1'],
                     'Ru':['[Kr] 4d7 5s1'],'Rh':['[Kr] 4d8 5s1'],'Pd':['[Kr] 4d10'],'Ag':['[Kr] 4d10 5s1'],'Cd':['[Kr] 4d10 5s2'],
                     'In':['[Kr] 4d10 5s2 5p1'],'Sn':['[Kr] 4d10 5s2 5p2'],'Sb':['[Kr] 4d10 5s2 5p3'],'Te':['[Kr] 4d10 5s2 5p4'],'Cs':['[Xe] 6s1'],'Ba':['[Xe] 6s2'],
                     'La':['[Xe] 5d1 6s2'],'Ce':['[Xe] 4f1 5d1 6s2'],'Hf':['[Xe] 4f14 5d2 6s2'],'Ta':['[Xe] 4f14 5d3 6s2'],
                     'W':['[Xe] 4f14 5d5 6s1'],'Re':['[Xe] 4f14 5d5 6s2'],'Os':['[Xe] 4f14 5d6 6s2'],
                     'Ir':['[Xe] 4f14 5d7 6s2'],'Pt':['[Xe] 4f14 5d10'],'Au':['[Xe] 4f14 5d10 6s1'],'Hg':['[Xe] 4f14 5d10 6s2'],
                     'Tl':['[Xe] 4f14 5d10 6s2 6p2'],'Pb':['[Xe] 4f14 5d10 6s2 6p2'],'Bi':['[Xe] 4f14 5d10 6s2 6p3'],
                     'Tc':['[Kr] 4d5 5s2'],'Fr':['[Rn]7s1'],'Ra':['[Rn]7s2'],'Pr':['[Xe]4f3 6s2'],
                     'Nd':['[Xe] 4f4 6s2'],'Pm':['[Xe] 4f5 6s2'],'Sm':['[Xe] 4f6 6s2'],
                     'Eu':['[Xe] 4f7 6s2'],'Gd':['[Xe] 4f7 5d1 6s2'],'Tb':['[Xe] 4f9 6s2'],
                     'Dy':['[Xe] 4f10 6s2'],'Ho':['[Xe] 4f11 6s2'],'Er':['[Xe] 4f12 6s2'],
                     'Tm':['[Xe] 4f13 6s2'],'Yb':['[Xe] 4f14 6s2'],'Lu':['[Xe] 4f14 5d1 6s2'],
                     'Po':['[Xe] 4f14 5d10 6s2 6p4'],'At':['[Xe] 4f14 5d10 6s2 6p5'],
                     'Ac':['[Rn] 6d1 7s2'],'Th':['[Rn] 6d2 7s2'],'Pa':['[Rn] 5f2 6d1 7s2'],
                     'U':['[Rn] 5f3 6d1 7s2'],'Np':['[Rn] 5f4 6d1 7s2'],'Pu':['[Rn] 5f6 7s2'],
                     'Am':['[Rn] 5f7 7s2'],'Cm':['[Rn] 5f7 6d1 7s2'],'Bk':['[Rn] 5f9 7s2'],
                     'Cf':['[Rn] 5f10 7s2'],'Es':['[Rn] 5f11 7s2'],'Fm':['[Rn] 5f12 7s2'],
                     'Md':['[Rn] 5f13 7s2'],'No':['[Rn] 5f14 7s2'],'Lr':['[Rn] 5f14 6d1 7s2'],
                     'Rf':['[Rn] 5f14 6d2 7s2'],'Db':['[Rn] 5f14 6d3 7s2'],
                     'Sg':['[Rn] 5f14 6d4 7s2'],'Bh':['[Rn] 5f14 6d5 7s2'],
                     'Hs':['[Rn] 5f14 6d6 7s2'],'Mt':['[Rn] 5f14 6d7 7s2'],'Xe': ['[Kr] 4d10 5s2 5p6'], 'He':['1s2'], 'Kr':['[Ar] 3d10 4s2 4p6'], 'Ar': ['[Ne] 3s2 3p6'], 'Ne':['[He] 2s2 2p6']}
        orbitals = {"s1":0,"s2":1,"p1":2,"p2":3,"p3":4,"p4":5,"p5":6,"p6":7,"d1":8,"d2":9,"d3":10,"d4":11,
            "d5":12,"d6":13,"d7":14,"d8":15,"d9":16,"d10":17,"f1":18,"f2":19,"f3":20,"f4":21,
            "f5":22,"f6":23,"f7":24,"f8":25,"f9":26,"f10":27,"f11":28,"f12":29,"f13":30,"f14":31}
        hv = np.zeros(shape=(32,1))
        hvs = {}
        for key in elements.keys():
            element = key
            hv = np.zeros(shape=(32,1))
            s = elements[key][0]
            sp = (re.split('(\s+)', s))
            if key == "H":
                hv[0] = 1
            if key != "H":
                for j in range(1,len(sp)):
                    if sp[j] != ' ':
                        n = sp[j][:1]
                        orb = sp[j][1:]
                        hv[orbitals[orb]] = 1
            hvs[element] = hv
        hot_fea = np.vstack([make_hot_for_atom_i(crystal,i,hvs) for i in range(len(crystal))])
        hot_fea = torch.from_numpy(hot_fea).float()
        hot_fea = torch.Tensor(hot_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:

                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return ([atom_fea, hot_fea], nbr_fea, nbr_fea_idx), target, cif_id
