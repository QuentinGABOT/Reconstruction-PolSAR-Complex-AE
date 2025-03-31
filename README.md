# Preserving polarimetric properties in PolSAR image reconstruction through Complex-Valued Auto-Encoders

This repository provides the inference code as well as pretrained networks for our paper.

To test inferences, you can proceed by

1- creating a virtual environment

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

2- Install the required dependencies

```
python -m pip install -r requirements.txt
```

3- Execute the training script

```
python -m torchtmpl.main config.yml train
```

And for testing using the provided trained model

```
python -m torchtmpl.main config.yml test logs/AutoEncoder
```
