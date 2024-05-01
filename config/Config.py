# config.py

class BaseModelConfig:
    # General parameters
    epochs = 100
    batch_size = 64
    learning_rate = 0.001

class CNNConfig(BaseModelConfig):
    # Specific parameters for CNN
    filter_sizes = [3, 4, 5]
    num_filters = 100
    dropout_rate = 0.5

class RNNConfig(BaseModelConfig):
    # Specific parameters for RNN
    hidden_units = 200
    num_layers = 2
    dropout_rate = 0.3

class BGRUConfig(BaseModelConfig):
    # Specific parameters for BGRU
    hidden_units = 150
    num_layers = 1
    dropout_rate = 0.2