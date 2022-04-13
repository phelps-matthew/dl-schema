# dl-schema
A deep learning training template

## Install
```
git clone https://github.com/phelps-matthew/dl-schema.git
cd dl-schema
pip install -e .
```

## Usage
* Download and extract MNIST data set
```python
python /data/create_mnist_dataset.py
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
python train.py --config_path train_cfg.yaml --lr 0.001
```
