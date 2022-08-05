# Orbital-Graph-Convolutional-Neural-Network
OGCNN
This is the repository for our work on property prediction for crystals. In this work we have used ideas from the Orbital Field matrix and Crystal Graph Convolutional Neural Network to predict material properties with a higher accuracy.

Paper link:https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.093801

# Important paper referenced 
The two important papers referenced for this work are:
1. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
2. Machine learning reveals orbital interaction in crystalline materials,  Science and Technology of Advanced Materials
Volume 18, 2017 - Issue 1.(https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1378060)

We used the ideas from these papers and did some of our modifications to develop the OGCNN which gives a higher performance than the seminal work of CGCNN

# Prerequisites
To run the OGCNN code the following packages are required
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org). It is preferable to install this package via pip
- [ase](https://wiki.fysik.dtu.dk/ase/)

It is advised to create a new conda environment and then install these packages. To create a new environment please refer to the conda documentation on managing environments (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Usage

To input crystal structures to OGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

The dataset that we use for this work are in the cif format. 

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The values of the target properties for each crystal in the dataset.

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications. The `atom_init.json` file has some of the basic atomic features encoded. Please refer the supplementary information of the paper to find out more about the basic atomic features.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There are two examples of customized datasets in the repository: `data/sample-regression` for regression and `data/sample-classification` for classification. 

**For advanced PyTorch users**
The above method of creating a customized dataset uses the `CIFData` class in `ogcnn.data`. If you want a more flexible way to input crystal structures and more feture descriptors to the model, PyTorch has a great [Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#sphx-glr-beginner-data-loading-tutorial-py) for writing your own dataset class.


### Train a OGCNN model

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in the directory that you choose to have main.py, you can train a OGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/sample-regression` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 8 --val-size 1 --test-size 1 data/sample-regression
```
or alternatively
```bash
python main.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 data/sample-regression
```

You can also train a classification model with label `--task classification`. For instance, you can use `data/sample-classification` by:

Although in the OGCNN work, we have not done any classification tasks. OGCNN similar to CGCNN has a switch to do the classification tasks which can run by using the following commands. 
```bash
python main.py --task classification --train-size 5 --val-size 2 --test-size 3 data/sample-classification
```

After training, you will get three files in the same directory as the main.py file.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained OGCNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a [pre-trained OGCNN model](pre-trained) named `pre-trained.pth.tar`.

Then, in directory where you have your predict.py script, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

For instance, you can predict the formation energies of the crystals in `data/sample-regression`:

```bash
python predict.py pre-trained/formation-energy-per-atom.pth.tar data/sample-regression
```

After predicting, you will get one file in `ogcnn` directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](data/material-data).

## Authors

This work was primarily done by Rishikesh Magar,Mohammadreza Karamad and Yuting Shi and was advised by Prof. Amir Barati Farimani, CMU







