import tensorflow as tf
from tensorflow import keras

from tcn import TCN


class Encoder(tf.keras.Model):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        num_stages: int = 1,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding: str = "same",
        weight_norm: bool = False,
        batch_norm: bool = False,
        layer_norm: bool = False,
        use_residual: bool = True,
        **kwargs
    ):
        """TCN Encoder
        The encoder consists of num_stages TCN blocks stacked on top of each
        other.
        Optionally, residual connections can be used between consecutive TCN stages.

        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilitation base
        :param dropout_rate: dropout rate
        :param num_stages: number of stages
        :param activation: the activation function used throughout the encoder
        :param kernel_initializer: the mode how the kernels are initialized
        :param padding: padding, usually' causal' or 'same'
        :param weight_norm: if weight normalization shall be used
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        :param use_residual: if residual connections shall be used between stacks. (
        Note: In the TCN-layer, residual connections are always used)
        :param kwargs:
        """
        super(Encoder, self).__init__(kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.num_stages = num_stages
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.use_residual = use_residual

        self.tcn_list = []
        for i in range(num_stages):
            self.tcn_list.append(
                TCN(
                    num_stages=2,
                    num_filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    dilation_base=self.dilation_base,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    final_activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    padding=self.padding,
                    weight_norm=self.weight_norm,
                    batch_norm=self.batch_norm,
                    layer_norm=self.layer_norm,
                    return_sequence=True,
                )
            )
        self.normalization_layers = []

        for i in range(num_stages - 1):
            self.normalization_layers.append(
                keras.layers.LayerNormalization(epsilon=1e-6)
            )

    def call(self, data_encoder):
        out_last = out = self.tcn_list[0](data_encoder)
        for i in range(self.num_stages - 1):
            out = self.tcn_list[i + 1](out)
            if self.use_residual:
                out = self.normalization_layers[i](out_last + out)
            else:
                out = self.normalization_layers[i](out)
        return out
