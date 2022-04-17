# dl-schema
A deep learning training template constructed as a minimal working MNIST example. Utilizes dataclasses as flexible train configs and mlflow for analytics and artifact logging.

## Install
```
git clone https://github.com/phelps-matthew/dl-schema.git
cd dl-schema
pip install -e .
```
### Dependencies
```
pip install -U tensorflow
pip install tensorflow-addons mlflow pyrallis pandas numpy tqdm pillow matplotlib
```

## Usage
* Download the MNIST data set, extract, and write to tfrecords
```python
cd data
python create_mnist_dataset.py
python write_tfrecords.py
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

