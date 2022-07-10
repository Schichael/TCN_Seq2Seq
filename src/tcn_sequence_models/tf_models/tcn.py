import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LayerNormalization
from tensorflow_addons.layers import WeightNormalization


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_stages: int,
        num_filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float,
        activation: str = "relu",
        final_activation: str = "relu",
        kernel_initializer: str = "he_normal",
        padding: str = "causal",
        weight_norm: bool = False,
        batch_norm: bool = False,
        layer_norm: bool = False,
        **kwargs,
    ):
        """The Residual Block

        :param num_stages: The number of convolutional layers.
        :param num_filters: The number of filters in a convolutional layer of the TCN.
        :param kernel_size: The size of every kernel in a convolutional layer.
        :param dilation_rate: The dilation power of 2 we are using for this
        residual block
        :param dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        :param activation: activation function
        :param kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        :param padding: the used padding
        :param weight_norm: Whether to use weight normalization in the residual
        layers or not
        :param batch_norm: Whether to use batch normalization in the residual
        layers or not.
        :param layer_norm: Whether to use layer normalization in the residual layers
        or not.
        :param kwargs: Any initializers for Layer class
        """

        super(ResidualBlock, self).__init__(**kwargs)

        self.num_stages = num_stages
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.layers = []
        self.dropout = None
        self.shape_match_conv = None
        self.final_activation = Activation(final_activation)

        for k in range(self.num_stages):
            name = f"conv1D_{k}"
            conv1D = Conv1D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                name=name,
                kernel_initializer=self.kernel_initializer,
            )
            if self.weight_norm:
                conv1D = WeightNormalization(conv1D)
            self.layers.append(conv1D)

            if self.batch_norm:
                self.layers.append(BatchNormalization())
            elif self.layer_norm:
                self.layers.append(LayerNormalization())

            if k < self.num_stages - 1:
                self.layers.append(Activation(self.activation))
                self.layers.append(Dropout(rate=self.dropout_rate))

    def build(self, input_shape):

        # Match input and output shapes
        if self.num_filters != input_shape[-1]:
            name = "matching_conv1D"
            self.shape_match_conv = Conv1D(
                filters=self.num_filters,
                kernel_size=1,
                padding="same",
                name=name,
                kernel_initializer=self.kernel_initializer,
            )
        else:
            self.shape_match_conv = Lambda(lambda x: x)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        x2 = self.shape_match_conv(inputs)
        res_x = layers.add([x2, x])
        res_act_x = self.final_activation(res_x)
        return res_act_x


class TCN(Model):
    def __init__(
        self,
        max_seq_len: int,
        num_stages: int,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        num_layers: int = None,
        activation: str = "relu",
        final_activation: str = "relu",
        kernel_initializer: str = "he_normal",
        padding: str = "causal",
        weight_norm: bool = False,
        batch_norm: bool = False,
        layer_norm: bool = False,
        return_sequence=False,
    ):
        """TCN stage. It uses as many layers as needed to get a connection from first
        input to last output or num_layers if specified.


        :param max_seq_len: maximum sequence length that is used to compute the
        number of layers
        :param num_stages: number of stages in the residual blocks
        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param activation: activation function used for CNNs
        :param final_activation: activation function used at hte end of each
        residual block
        :param kernel_initializer: the mode how the kernels are initialized
        :param padding: padding, usually' causal' or 'same'
        :param weight_norm: if weight normalization shall be used
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        :param return_sequence: if the output sequence shall be returned or just the
        last output
        """
        super(TCN, self).__init__()
        self.max_seq_len = max_seq_len
        self.num_stages = num_stages
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_layers = num_layers

        self.res_block_list = []
        self.input_chunk_length = None
        self.return_sequence = return_sequence
        self.dilation_rates = []

        if self.batch_norm + self.layer_norm + self.weight_norm > 1:
            raise ValueError("Only one normalization can be specified at once.")

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (
            self.dilation_base ** self.num_layers
        ) / (self.dilation_base - 1)

    def build(self, input_shape):
        if self.num_layers is None:
            self.num_layers = math.ceil(
                math.log(
                    (
                        (self.max_seq_len - 1)
                        * (self.dilation_base - 1)
                        / (self.kernel_size - 1)
                        / self.num_stages
                        + 1
                    ),
                    self.dilation_base,
                )
            )

        for i in range(self.num_layers):
            dilation_rate = self.dilation_base ** i
            self.dilation_rates.append(dilation_rate)
            self.res_block_list.append(
                ResidualBlock(
                    self.num_stages,
                    self.num_filters,
                    self.kernel_size,
                    dilation_rate,
                    self.dropout_rate,
                    self.activation,
                    self.final_activation,
                    self.kernel_initializer,
                    self.padding,
                    self.weight_norm,
                    self.batch_norm,
                    self.layer_norm,
                )
            )

    def call(self, inputs, training=None):

        x = inputs
        for res_block in self.res_block_list:
            x = res_block(x, training=training)
        if self.return_sequence:
            return x
        else:
            x_last = x[:, -1, :]
            return x_last
