# Crystal Twins 
This is the code repository for the work the work Crystal Twins: Self-supervised Learning for Crystalline Material Property Prediction. In this work we develop a SSL framework for material property prediction.

We developed a framework for SSL learning for crystalline materials. Our method is based on the Barlow Twins Loss Function and the SimSimaese Network. 

![image](https://user-images.githubusercontent.com/43094762/200387436-79b62495-dcbe-4465-b071-6e999bc66c45.png)

## Benchmark
The CGCNN model we pretrained using SSL has been tested on the Matbench datasets and the some of the databases from Materials Project. We have made the pretrained models available for general use. The model was pretrained on the Matminer database and the hMOF database. In total, we had 428K crystalline materials for training the ML model.

# Prerequisites
To run the CT code the following packages are required
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org). 
- [ase](https://wiki.fysik.dtu.dk/ase/)

It is advised to create a new conda environment and then install these packages. To create a new environment please refer to the conda documentation on managing environments (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Usage

To input crystal structures to OGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

The dataset that we use for this work are in the cif format. 

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The values of the target properties for each crystal in the datase

You can create a customized dataset by creating a directory `root_dir` with the following files: 
<!-- 
1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)
 -->
1. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. The `atom_init.json` file has some of the basic atomic features encoded. Please refer the supplementary information of the paper to find out more about the basic atomic features.

2. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal

# Pretrained model:
The models have been pretrained on the cif files in the hMOF database and the matminer database. In total we aggregate 428,275 cif files for the pretraining. The pretrained models along along with their config files are made available in the repository. We have two pretrained models using the Barlow Twins and the SimSiamese loss functions. 
To run the pretrained model run the command 

```bash
 python contrast.py
 ```
The parameters for the pretraining of model can be modified in the `config.yaml`

# Finetuning model:
For Finetuning the model, we initialize with the pre-trained weights and finetune it for the downstream task. To train it on the matbench datasets, the run the finetuning run 
For your own dataset, you need to have run `id_prop.csv` in the dataset folder.
```bash
 python finetune_cgcnn.py
 ``` 
 To run the model on the matbench benchmark run:
 ```bash
 python finetune_matbench.py
 ```  
# Data
The matbench data is available at - [Matbench](https://matbench.materialsproject.org)

For datasets in the Materials Project - [MP](https://materialsproject.org). 

# Acknowledgements

- CGCNN: [https://github.com/txie-93/cgcnn](https://github.com/txie-93/cgcnn)
- Barlow Twins [https://github.com/facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)
- SimSiam [https://github.com/facebookresearch/simsiam](https://github.com/facebookresearch/simsiam)

