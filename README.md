# &#127796; dl-schema
A deep learning training template constructed as a minimal working MNIST example. Utilizes dataclasses as flexible train configs and mlflow for analytics and artifact logging.

## Install
```
git clone https://github.com/phelps-matthew/dl-schema.git
cd dl-schema
pip install -e .
```
### Dependencies
```
pip install -U torch torchvision
pip install mlflow pyrallis pandas numpy tqdm pillow matplotlib
```

## Usage
* Download and extrac the MNIST dataset
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

