# Deep Graph Attention Neural Network (DGANN) 

![](<p align="center">< img height="300" width="700" src="https://github.com/coffee19850519/PPDocking/blob/master/figure/FIG1.jpg"></p>)

## 1. Installation
Start by grabbing this source codes:
```
git clone https://github.com/coffee19850519/PPDocking
```

### Environment
Use python virutal environment with conda.
```
conda create -n PPDocking python=3.7
conda activate PPDocking
```

Then install all python packages in bash.
```
pip install -r requirements.txt
```

## 2. Examples
### Requirements for preparing data:

 - Use the following file structure
      ```
      raw_data/
      |__pdb/
      |__pssm/
      |__node_feature/
      |__caseID.lst
      ```
   The `pdb` folder contains the PDB files of docking models, `pssm` contains the PSSM files, and `node_feature` contains the graph features files.
   The `caseID.lst` is a list of class ID and PDB file name for each docking model, like `7CEI`.
   
 - [Biopython](https://biopython.org/) can be used to process PDB file.
   [PSSMGen](https://github.com/DeepRank/PSSMGen) can be used to get consistent PSSM and PDB files. 
   There are already installed along with PPDock.

### Use our trained model

You can directly use our trained model to score your docking conformations. 
You can find the model in the package subfolder `/example/model/0fold_classification_model.pt`.
You can find the data in the package subfolder `/example/data/0_classification_test.pt`.

#### Quick start
You can use data and model we prepared in `example/data` and `example/model` folder with type:
```
python import_data.py
```


#### Prepare datasets
You should go into your `pssm_generating` folder and run `compute_pssm.py` to generate side chain information first.
```
python compute_pssm.py
```
Then you should go into your `feature_extraction` folder and run `generate_node_feature.py` to generate side chain information.
```
python generate_node_feature.py
```

Finally, you can use the model we prepared `example/model` folder with type:
```
python import_data.py
```

## 3. Citation
If you use iScore software, please cite the following articles:
```
tex code
```