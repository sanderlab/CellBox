[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dfci/CellBox/version_for_initial_manuscript)

# Quick Start

Easily try PertBio online with Binder 

1. Go to: https://mybinder.org/v2/gh/dfci/CellBox/version_for_initial_manuscript
2. From the New dropdown, click Terminal 
3. Run the following command for a short example of model training process: 

```
python scripts/main.py -config=configs/example.cfg.json
```

Alternatively, in project folder, do the same command

# Installation

## Install using pip 
The following command will install pertbio from a particular branch using the '@' notation:

```
pip install git+https://github.com/dfci/CellBox.git@version_for_initial_manuscript#egg=pertbio\&subdirectory=pertbio
```

## Install using setup.py
Clone repository and in the pertbio folder run:

```
python3.6 setup.py install
```

Only python3.6 supported. Anaconda or pipenv is recommended to create python environment. 

Now you can test if the installation is successful

```
import pertbio
pertbio.VERSION
```

# Project Structure

## Data files: in ./data/ folder
* _node_index.txt_: names of each protein/phenotypic node.
* _expr_index.txt_: information each perturbation condition (also see loo_label.csv).
* _expr.csv_: Protein expression data from RPPA for the protein nodes and phenotypic node values. Each row is a condition while each column is a node.
* _pert.csv_: Perturbation strength and target of all perturbation conditions. Used as input for differential equations.

## pertbio package:
* _CellBox_ is defined in model.py
* dataset factory for random parition and leave one out tasks
* some training util functions in tensorflow

## One click model construction

### __Step 1: Create experiment json files (some examples can be found under ./configs/)__
* Make sure to specify the experiment_id and experiment_type
	* experiment_id: name of the experiments, would be used to generate results folders
	* experiment_type: currently available tasks are {"random partition", "leave one out (w/o single)", "leave one out (w/ single)", "full data", "single to combo"]}
* Different training stages can be specified using stages and sub_stages

### __Step 2: Use main.py to construct models using random partition of dataset__

The experiment type configuration file is specified by --experiment_config_path OR -config

```
python scripts/main.py -config=configs/random_partition.cfg.json
```

Note: always run the script in the root folder.


A random seed can also be assigned by using argument --working_index OR -i

```
python scripts/main.py -config=configs/random_partition.cfg.json -i=1234
```


When training with leave-one-out validation, make sure to specify the drug index --drug_index OR -drug to leave out from training.


### __Step 3: Analyze result files__
* You should see a experiment folder generated under results using the date and experiment_id.
* Under experiment folder, you would see different models run with different random seeds
* Under each model folder, you would have:
	* _record_eval.csv_: log file with loss changes and time used.
	* _random_pos.csv_: how the data was split (only for random partitions)
	* _best.W_, _best.alpha_, _best.eps_: model parameters snapshot for each training stage
	* predicted training set .csv, predicted nodes of training set (average prediction over last 20% of ODE simulation, time derivative at end point, max-min over last 20% ODE simulation)
	* _best.test_hat_: Prediction on test set, using the best model for each stage
	* _.ckpt_ files are the final models in tensorflow compatible format.
