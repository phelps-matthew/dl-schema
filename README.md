# &#127796; dl-schema
A deep learning training template constructed as a minimal working MNIST example. Utilizes dataclasses as flexible train configs and mlflow for analytics and artifact logging.

## Install
```
# create `dl` conda environment
conda create -n dl python=3.10 pip
conda activate dl

# install torch and dependencies, assumes cuda version >= 11.0
pip install torch=2.0.0 torchvision=0.15.1
pip install mlflow=2.2.2 pyrallis=0.3.1 
pip install pandas tqdm pillow matplotlib 

# install hyperparameter search dependencies
pip install ray[tune] hyperopt

# install dl-schema repo
git clone https://github.com/phelps-matthew/dl-schema.git
cd dl-schema
pip install -e .
```

## Usage
* Download and extract the MNIST dataset
```python
cd dl-schema
python data/create_mnist_dataset.py
```
* Train small CNN model (ResNet-18)
```python
python train.py
```
* View train configuration options
```python
python train.py --help
```
* Train from yaml configuration, with CLI override
```python
python train.py --config_path configs/resnet.yaml --lr 0.001 --gpus [7]
```
* Start mlflow ui to visualize results
```
# navgiate to dl_schema root directory containing `mlruns`
mlflow ui
# to set host and port
mlflow ui --host 0.0.0.0 --port 8080
```
* Serialize dataclass train config to yaml, outputting `configs/train_cfg.yaml`
```python
python cfg.py
```

## Hyperparameter Experiments
* Use ray tune to perform multi-gpu hyperparameter search
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python tune.py --exp_name hyper_search
```
