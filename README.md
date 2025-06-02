# Preserving polarimetric properties in PolSAR image reconstruction through Complex-Valued Auto-Encoders

This repository provides the inference code as well as pretrained networks for our paper.

To test inferences, you can proceed by

1- creating a virtual environment

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .

```
2- Dataset installation

For running the code, you need to install the library and provide a
configuration file (we provide a designated configuration file in the configs folder).

A path to the dataset must be provided in the configuration file: we suggest to specify the directory of the unzipped PolSF dataset.

To download the PolSF dataset, please follow the following link: https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip. Warning: you must move the file SF-ALOS2-label2d.png from the sub-folder Config_labels to the main folder.


3- Execute the training script

```
python -m torchtmpl.main config.yml train
```

And for testing using the provided trained model

```
python -m torchtmpl.main config.yml test logs/AutoEncoder
```
