# Backbone Architecture
# from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, BatchNormalization, Dropout, MaxPooling1D, Reshape
from tensorflow.keras import Sequential

class LinearBackbone(Layer):
    def __init__(self, output_dim, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.n_layers = n_layers
        # print(f"{output_dim = }")
    
    def build(self, input_shape):
        self.layers = []
        for i in range(self.n_layers):
            layers_out_dim = self.output_dim + (self.n_layers - (i + 1))*((input_shape[-1] - self.output_dim)//self.n_layers)
            # print(layers_out_dim)
            self.layers.append(Dense(layers_out_dim,  activation='relu'))

    def call(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
            # print(x.shape)
        return x
    
    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'n_layers': self.n_layers,
            })
        return config  
  
# ConvNet (input: 1 x 60 x n_factors):
# 1 x n_factors 2d kernel x 8; output: (8 x 60 x1) -> (8 x 60)
# 5 1d kernel x 8; output: (8 x 56)
# 2 max pooling; output: (8 x 28)
# batch_norm
# dropout
# 3 1d kernel x 8; output: (8 x 24)
# 2 max pooling; output: (8 x 12)
# batch_norm
# dropout
# transformer (reshape to (12 x 8)

class ConvBackbone(Layer):
    def __init__(self, output_dim, dur_kernels, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dur_kernels = dur_kernels
        self.dropout = dropout

    def build(self, input_shape):
        """_summary_

        Args:
            input_shape (tuple): shape should be None * seq_len * num_factors
        """
        # print(f"{input_shape = }")

        # (None, seq_len, num_factors, 1) -> (None, seq_len, 1, 4)
        self.factor_conv = Conv2D(
            filters = self.output_dim,
            kernel_size = (1, input_shape[-1]),
            # data_format = 'channels_first',
            kernel_regularizer = 'l1', #! Imlemented L1 reg
            input_shape = (None, 1) + input_shape[-2:],
            activation = 'relu'
        )

        # (None, seq_len, output_dim) -> (None, ..., output_dim)
        self.dur_conv_layers = []
        for i, k_size in enumerate(self.dur_kernels):
            module = Sequential()
            module.add(Conv1D(
                filters = self.output_dim,
                kernel_size = k_size,
                activation = 'relu'
                # data_format = 'channels_first'
            ))
            module.add(MaxPooling1D(
                pool_size=2, 
                # data_format = 'channels_first'
            ))
            module.add(BatchNormalization())    
            module.add(Dropout(self.dropout))    
            self.dur_conv_layers.append(module)

    def call(self, x):
        
        # print(f"{x.shape = }") # (None, seq_len, num_factors)

        x = tf.expand_dims(x, axis=-1)
        # print(f"{x.shape = }") # (None, seq_len, num_factors, 1)

        x = self.factor_conv(x)
        # print(f"{x.shape = }") # (None, seq_len, 1, output_dim)

        x = tf.squeeze(x, axis=-2)
        # print(f"{x.shape = }") # (None, seq_len, output_dim)

        for dur_conv_layer in self.dur_conv_layers:
            x = dur_conv_layer(x)
            # print(f"{x.shape = }") # (None, ..., output_dim)

        return x

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'dur_kernels': self.dur_kernels,
            'dropout': self.dropout,
            })
        return config  

def get_backbone(type: str, output_dim):
    if type=='linear':
        return LinearBackbone(output_dim, 2)
    elif type=='conv':
        return ConvBackbone(output_dim, [3], dropout=0.2)
    return lambda x: x