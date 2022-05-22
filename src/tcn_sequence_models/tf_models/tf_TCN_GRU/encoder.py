import tensorflow as tf
from tensorflow import keras

from tcn_sequence_models.tf_models.tcn import TCN


class Encoder(tf.keras.Model):
    def __init__(
        self,
        max_seq_len: int,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        num_stages: int = 1,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding: str = "same",
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        """TCN Encoder stage
        The encoder consists of num_stages TCN blocks stacked on top of each
        other.

        :param max_seq_len: maximum sequence length that is used to compute the
        number of layers
        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param num_stages: number of stages
        :param activation: the activation function used throughout the encoder
        :param kernel_initializer: the mode how the kernels are initialized
        :param padding: padding, usually' causal' or 'same'
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        """
        super(Encoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.num_stages = num_stages
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        self.tcn_list = []

        for i in range(self.num_stages):
            self.tcn_list.append(
                TCN(
                    max_seq_len=self.max_seq_len // 2,
                    num_stages=2,
                    num_filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    dilation_base=self.dilation_base,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    final_activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    padding=self.padding,
                    weight_norm=False,
                    batch_norm=self.batch_norm,
                    layer_norm=self.layer_norm,
                    return_sequence=True,
                )
            )

        # Normalization after each TCN stage
        self.normalization_layers = []

        for i in range(num_stages - 1):
            if self.layer_norm:
                self.normalization_layers.append(
                    keras.layers.LayerNormalization(epsilon=1e-6)
                )
            else:
                self.normalization_layers.append(keras.layers.BatchNormalization())

    @tf.function
    def call(self, data_encoder, training=None):

        out = self.tcn_list[0](data_encoder, training=training)
        for i in range(self.num_stages - 1):
            out = self.tcn_list[i + 1](out, training=training)
        return out
