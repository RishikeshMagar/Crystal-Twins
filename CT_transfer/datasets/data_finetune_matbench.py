from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import pandas as pd
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from matbench.bench import MatbenchBenchmark
# from augmentation import RotationTransformation, PerturbStructureTransformation, SwapAxesTransformation


def get_train_val_test_loader(train_dataset, test_dataset, collate_fn=default_collate,
                              batch_size=64, 
                              return_test=False, num_workers=1, pin_memory=False, 
                              **kwargs):
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
    total_train_size = len(train_dataset)
    total_test_size  = len(test_dataset)
    # if train_ratio is None:
    #     #assert val_ratio + test_ratio < 1
    #     train_ratio = 1 - val_ratio #- test_ratio 
    # #     # print('[Warning] train_ratio is None, using all training data.')
    # else:
    #     assert train_ratio + val_ratio + test_ratio <= 1
    # train_ratio = 1 - val_ratio - test_ratio
    # indices = list(range(total_size))
    # print("The random seed is: ", random_seed)
    # np.random.seed(random_seed)
    # np.random.shuffle(indices)
    # if kwargs['train_size']:
    #     train_size = kwargs['train_size']
    # else:
    #     train_size = int(train_ratio * total_size)
    # if kwargs['test_size']:
    #     test_size = kwargs['test_size']
    # else:
    #     test_size = int(test_ratio * total_size)
    # if kwargs['val_size']:
    #     valid_size = kwargs['val_size']
    # else:
    #     valid_size = int(val_ratio * total_size)

    valid_size = len(test_dataset)
    test_size = len(test_dataset)

    train_size = len(train_dataset) - valid_size

    print('Train size: {}, Validation size: {}, Test size: {}'.format(
        train_size, valid_size, test_size
    ))
    
    indices_train_val =  list(range(len(train_dataset)))
    train_sampler = SubsetRandomSampler(indices_train_val[:train_size])
    
    val_sampler = SubsetRandomSampler(
        indices_train_val[-valid_size:])
    
    # if return_test:
    #     test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory, drop_last = True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory, drop_last = True)
    if return_test:
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
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
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
            torch.stack(batch_target, dim=0),\
            batch_cif_ids


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
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIF_train_val_Data(Dataset):
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
    def __init__(self,task_type, subset_data, fold_num, max_num_nbr=12, radius=8, dmin=0, step=0.2):
        
        self.mb = MatbenchBenchmark(autoload=False,subset=[subset_data])
        
        for task in self.mb.tasks:
            task.load()
            train_df = task.get_train_and_val_data(fold_num, as_type="df")
        # self.root_dir = root_dir
        self.task_type = task_type
        self.max_num_nbr, self.radius = max_num_nbr, radius
        # assert os.path.exists(root_dir), 'root_dir does not exist!'
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        self.id_prop_data = train_df
        # random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join('datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        # self.rotater = RotationTransformation()
        # self.perturber = PerturbStructureTransformation(distance=0.05, min_distance=0.0)        

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        crystal, target = self.id_prop_data.iloc[idx]
        cif_id = self.id_prop_data.iloc[idx].name
        # crystal = Structure.from_file(os.path.join(self.root_dir,
        #                                            cif_id))

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
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
        
        if self.task_type == 'regression':
            target = torch.Tensor([float(target)])
        else:
            if target == 'False' or target == 0:
                label = 0
            elif target == 'True' or target == 1:
                label = 1
            target = torch.LongTensor([(label)])
        
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


class CIF_test_Data(Dataset):
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
    def __init__(self,task_type, subset_data, fold_num, max_num_nbr=12, radius=8, dmin=0, step=0.2):
        
        self.mb_test = MatbenchBenchmark(autoload=False,subset=[subset_data])
        
        for task in self.mb_test.tasks:
            task.load()
            test_df = task.get_test_data(fold_num, include_target=True, as_type="df")
        # self.root_dir = root_dir
        self.task_type = task_type
        self.max_num_nbr, self.radius = max_num_nbr, radius
        # assert os.path.exists(root_dir), 'root_dir does not exist!'
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        self.id_prop_test_data = test_df
        # random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join('datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        # self.rotater = RotationTransformation()
        # self.perturber = PerturbStructureTransformation(distance=0.05, min_distance=0.0)        

    def __len__(self):
        return len(self.id_prop_test_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        crystal, target = self.id_prop_test_data.iloc[idx]
        cif_id = self.id_prop_test_data.iloc[0].name
        # crystal = Structure.from_file(os.path.join(self.root_dir,
        #                                            cif_id))

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
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
        if self.task_type == 'regression':
            target = torch.Tensor([float(target)])
        else:
            if target == 'False' or target == 0:
                label = 0
            elif target == 'True' or target == 1:
                label = 1
            target = torch.LongTensor([(label)])
        
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
































        # crystal = crys.copy()
        # crystal_per_1 =  self.perturber.apply_transformation(crystal)
        # crystal_per_2 =  self.perturber.apply_transformation(crystal)

        # for i in range(3):
        #     axis = np.zeros(3)
        #     axis[i] = 1
        #     rot_ang = np.random.uniform(-90.0, 90.0)
        #     crystal_rot_1 = self.rotater.apply_transformation(crystal_per_1, axis, rot_ang, angle_in_radians=False)

        # for i in range(3):
        #     axis = np.zeros(3)
        #     axis[i] = 1
        #     rot_ang = np.random.uniform(-90.0, 90.0)
        #     crystal_rot_2 = self.rotater.apply_transformation(crystal_per_2, axis, rot_ang, angle_in_radians=False)


        # atom_fea_rot_1 = np.vstack([self.ari.get_atom_fea(crystal_rot_1[i].specie.number)
        #                       for i in range(len(crystal_rot_1))])
        # atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        # all_nbrs_rot_1 = crystal_rot_1.get_all_neighbors(self.radius, include_index=True)
        # all_nbrs_rot_1 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_1]
        # nbr_fea_idx_rot_1, nbr_fea_rot_1 = [], []
        # for nbr in all_nbrs_rot_1:
        #     if len(nbr) < self.max_num_nbr:
        #         warnings.warn('{} not find enough neighbors to build graph. '
        #                       'If it happens frequently, consider increase '
        #                       'radius.'.format(cif_id))
        #         nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
        #                            [0] * (self.max_num_nbr - len(nbr)))
        #         nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
        #                        [self.radius + 1.] * (self.max_num_nbr -
        #                                              len(nbr)))
        #     else:
        #         nbr_fea_idx_rot_1.append(list(map(lambda x: x[2],
        #                                     nbr[:self.max_num_nbr])))
        #         nbr_fea_rot_1.append(list(map(lambda x: x[1],
        #                                 nbr[:self.max_num_nbr])))
        # nbr_fea_idx_rot_1, nbr_fea_rot_1 = np.array(nbr_fea_idx_rot_1), np.array(nbr_fea_rot_1)
        # nbr_fea_rot_1 = self.gdf.expand(nbr_fea_rot_1)
        # atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        # nbr_fea_rot_1 = torch.Tensor(nbr_fea_rot_1)
        # nbr_fea_idx_rot_1 = torch.LongTensor(nbr_fea_idx_rot_1)
        # target = torch.Tensor([float(target)])

        # atom_fea_rot_2 = np.vstack([self.ari.get_atom_fea(crystal_rot_2[i].specie.number)
        #                       for i in range(len(crystal_rot_2))])
        # atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        # all_nbrs_rot_2 = crystal_rot_2.get_all_neighbors(self.radius, include_index=True)
        # all_nbrs_rot_2 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_2]
        # nbr_fea_idx_rot_2, nbr_fea_rot_2 = [], []
        # for nbr in all_nbrs_rot_2:
        #     if len(nbr) < self.max_num_nbr:
        #         warnings.warn('{} not find enough neighbors to build graph. '
        #                       'If it happens frequently, consider increase '
        #                       'radius.'.format(cif_id))
        #         nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
        #                            [0] * (self.max_num_nbr - len(nbr)))
        #         nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
        #                        [self.radius + 1.] * (self.max_num_nbr -
        #                                              len(nbr)))
        #     else:
        #         nbr_fea_idx_rot_2.append(list(map(lambda x: x[2],
        #                                     nbr[:self.max_num_nbr])))
        #         nbr_fea_rot_2.append(list(map(lambda x: x[1],
        #                                 nbr[:self.max_num_nbr])))
        # nbr_fea_idx_rot_2, nbr_fea_rot_2 = np.array(nbr_fea_idx_rot_2), np.array(nbr_fea_rot_2)
        # nbr_fea_rot_2 = self.gdf.expand(nbr_fea_rot_2)
        # atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        # nbr_fea_rot_2 = torch.Tensor(nbr_fea_rot_2)
        # nbr_fea_idx_rot_2 = torch.LongTensor(nbr_fea_idx_rot_2)
        # # target = torch.Tensor([float(target)])
        # # target = torch.Tensor([float(target)])
        # return (atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2),target, cif_id
