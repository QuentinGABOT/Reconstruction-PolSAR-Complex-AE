# Preserving polarimetric properties in PolSAR image reconstruction through Complex-Valued Auto-Encoders

This repository provides the inference code as well as pretrained networks for our paper.

```bibtex
@inproceedings{gabot:hal-04785702,
  TITLE = {{Preserving polarimetric properties in PolSAR image reconstruction through Complex-Valued Auto-Encoders}},
  AUTHOR = {Gabot, Quentin and Fix, J{\'e}r{\'e}my and Frontera-Pons, Joana and Ren, Chengfang and Ovarlez, Jean-Philippe},
  URL = {https://hal.science/hal-04785702},
  BOOKTITLE = {{RADAR 2024}},
  ADDRESS = {Rennes, France},
  YEAR = {2024},
  MONTH = Oct,
  KEYWORDS = {Complex-valued Auto-Encoders ; PolSAR image reconstruction ; polarimetric decompositions},
  PDF = {https://hal.science/hal-04785702v1/file/DEMR2024-056.pdf},
  HAL_ID = {hal-04785702},
  HAL_VERSION = {v1},
}
```

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
