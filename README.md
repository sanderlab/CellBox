<a target="_blank" href="https://colab.research.google.com/github/sanderlab/CellBox/blob/master/notebooks/cellbox_example_tf2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sanderlab/CellBox/9d13f3354f8b14bd896de6c8aa5db0b97c65ad12)

# CellBox

## Abstract
Systematic perturbation of cells followed by comprehensive measurements of molecular and phenotypic responses provides informative data resources for constructing computational models of cell biology. Models that generalize well beyond training data can be used to identify combinatorial perturbations of potential therapeutic interest. Major challenges for machine learning on large biological datasets are to find global optima in a complex multi-dimensional space and mechanistically interpret the solutions. To address these challenges, we introduce a hybrid approach that combines explicit mathematical models of cell dynamics with a machine learning framework, implemented in TensorFlow. We tested the modeling framework on a perturbation-response dataset of a melanoma cell line after drug treatments. The models can be efficiently trained to describe cellular behavior accurately. Even though completely data-driven and independent of prior knowledge, the resulting de novo network models recapitulate some known interactions. The approach is readily applicable to various kinetic models of cell biology. 

<p align="center">
	<img src="https://lh3.googleusercontent.com/d/15Lildcx8sC4shTalODLXqfibJTbnxmun=w600">
</p>

## Citation and Correspondence

This is CellBox scripts developed in Sander lab for the paper in _[Cell Systems](https://www.cell.com/cell-systems/pdfExtended/S2405-4712(20)30464-6)_ or [bioRxiv](https://www.biorxiv.org/content/10.1101/746842v3).

>Yuan, B.*, Shen, C.*, Luna, A., Korkut, A., Marks, D., Ingraham, J., Sander, C. CellBox: Interpretable Machine Learning for Perturbation Biology with Application to the Design of Cancer Combination Therapy. _Cell Systems_, 2020. 

Maintained by Bo Yuan, Judy Shen, and Augustin Luna.

If you want to discuss the usage or to report a bug, please use the 'Issues' function here on GitHub.

If you find `CellBox` useful for your research, please consider citing the corresponding publication.

For more information, please find our contact information [here](https://www.sanderlab.org/#/).

# Quick Start

Easily try `CellBox` online with Binder 

1. Go to: https://mybinder.org/v2/gh/sanderlab/CellBox/9d13f3354f8b14bd896de6c8aa5db0b97c65ad12
2. From the New dropdown, click Terminal 
3. Run the following command for a short example of model training process: 

```
python scripts/main.py -config=configs/Example.random_partition.json
```

Alternatively, in project folder, do the same command

# Installation

## Install using pip 
The following command will install cellbox from a particular branch using the '@' notation:

```
pip install git+https://github.com/dfci/CellBox.git@cell_systems_final#egg=cellbox\&subdirectory=cellbox
```

## Install using setup.py
Clone repository and in the `cellbox` folder run:

```
python3.6 setup.py install
```

Only python3.6 supported. Anaconda or pipenv is recommended to create python environment. 

Now you can test if the installation is successful

```
import cellbox
cellbox.VERSION
```

# Project Structure

## Data files: in ./data/ folder in GitHub repo used for example
* `node_index.txt`: names of each protein/phenotypic node.
* `expr_index.txt`: information each perturbation condition. This is one of the original data files we downloaded from [paper](https://elifesciences.org/articles/04640) and is only used here as a reference for the condition names. In other words the 2nd and 3rd columns are not being used in CellBox. See `loo_label.csv` for the actual indexing of perturbation targets.
* `expr.csv`: Protein expression data from RPPA for the protein nodes and phenotypic node values. Each row is a condition while each column is a node.
* `pert.csv`: Perturbation strength and target of all perturbation conditions. Used as input for differential equations.

## cellbox package:
* `CellBox` is defined in model.py
* A dataset factory function for random parition and leave one out tasks
* Some training util functions in tensorflow

## One click model construction

### __Step 1: Create experiment json files (some examples can be found under ./configs/)__
* Make sure to specify the experiment_id and experiment_type
	* `experiment_id`: name of the experiments, would be used to generate results folders
	* `experiment_type`: currently available tasks are {"random partition", "leave one out (w/o single)", "leave one out (w/ single)", "full data", "single to combo"]}
* Different training stages can be specified using `stages` and `sub_stages` in config file

### __Step 2: Use main.py to construct models using random partition of dataset__

The experiment type configuration file is specified by `--experiment_config_path` or `-config`

```
python scripts/main.py -config=configs/Example.random_partition.json
```

Note: always run the script in the root folder.


A random seed can also be assigned by using argument `--working_index` or `-i`

```
python scripts/main.py -config=configs/Example.random_partition.json -i=1234
```


When training with leave-one-out validation, make sure to specify the drug index `--drug_index` or `-drug` to leave out from training.


### __Step 3: Analyze result files__
* You should see a experiment folder generated under results using the date and `experiment_id`.
* Under experiment folder, you would see different models run with different random seeds
* Under each model folder, you would have:
	* `record_eval.csv`: log file with loss changes and time used.
	* `random_pos.csv`: how the data was split (only for random partitions)
	* `best.W`, `best.alpha`, `best.eps`: model parameters snapshot for each training stage
	* `best.test_hat`: Prediction on test set, using the best model for each stage
	* `.ckpt` files are the final models in tensorflow compatible format.
