# &#127796; dl-schema
A deep learning training template constructed as a minimal working MNIST example. Utilizes dataclasses as flexible train configs and mlflow for analytics and artifact logging.

## Install
```
# create `schema` conda environment
conda create -n schema python=3.9 pip
conda activate schema

# install torch and dependencies, assumes cuda version >= 11.0
pip install -U pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install mlflow pyrallis pandas tqdm pillow

# install dl-schema repo
git clone https://github.com/phelps-matthew/dl-schema.git
cd dl-schema
pip install -e .
```

## Usage
* Download and extract the MNIST dataset
```python
cd data
python create_mnist_dataset.py
```
* Train small CNN model
```python
python train.py
```
* View train configuration options
```python
python train.py --help
```
* Train from yaml configuration, with CLI override
```python
python train.py --config_path train_cfg.yaml --lr 0.001 --gpus [7]
```
* Start mlflow ui to visualize results
```
# navgiate to dl_schema root directory containing `mlruns`
mlflow ui
# to set host and port
mlflow ui --host 0.0.0.0 --port 8080
```
* Serialize dataclass train config to yaml, outputting `train_cfg.yaml`
```python
python cfg.py
```

