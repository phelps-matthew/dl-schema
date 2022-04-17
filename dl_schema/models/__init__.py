"""
Method for registering and building any models in ./models/*.

E.g., importing BabyCNN as below places in global scope for build_model function.
"""
from dl_schema.models.babycnn import BabyCNN
import tensorflow as tf


def build_model(model_class, cfg):
    """initialize model from class and model cfg. class must be in global scope"""
    class_ = globals()[model_class]
    model = tf.keras.models.Sequential([class_(cfg)])
    return model


if __name__ == "__main__":
    """Test the model"""
    from dl_schema.dataset import mnist_dataset
    from dl_schema.cfg import TrainConfig

    cfg = TrainConfig()
    cfg.data.bs = 2
    model = build_model(cfg.model.model_class, cfg.model)
    train_dataset = mnist_dataset(cfg)
    x, y = next(iter(train_dataset))
    print(model(x))
    print(model.summary())
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    print(model(x))
