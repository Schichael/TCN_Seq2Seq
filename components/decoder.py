import tensorflow as tf
from tensorflow import keras

from tcn import TCN
from multi_head_attention import MultiHeadAttention


class Decoder(tf.keras.Model):
    def __init__(
        self,
        num_stages: int,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        key_size: int,
        value_size: int,
        num_attention_heads: int,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding: str = "causal",
        weight_norm: bool = False,
        batch_norm: bool = False,
        layer_norm: bool = False,
        use_residual: bool = True,
        **kwargs
    ):
        """TCN Decoder
        The decoder consists of num_stages stages.
        Each stage begins with a TCN block. It follows a Encoder-Decoder
        Multi-Headed-Attention layer that connects the output of the encoder
        with the output of the TCN block. The output of the TCN block and the
        attention layer are summed and normalized.
        Optionally, residual connections can be used between the output of the last
        stage and the output of the TCN block of the next stage.
        After the last stage, another TCN block is added that generates the output.
        Note: the output is NOT the prediction. After the decoder, an output stage (
        e.g. a feed forward NN) still needs to be added to compute the prediction


        :param num_stages: number of stages
        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilitation base
        :param dropout_rate: dropout rate
        :param key_size: dimensionality of key/query
        :param value_size: dimensionality of value
        :param num_attention_heads: Number of attention heads to be used
        :param activation: the activation function used throughout the decoder
        :param kernel_initializer: the mode how the kernels are initialized
        :param padding: padding, usually' causal' or 'same'
        :param weight_norm: if weight normalization shall be used
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        :param use_residual: if residual connections shall be used between stacks. (
        Note: In the TCN-layer, residual connections are always used)
        :param kwargs:
        """
        super(Decoder, self).__init__(kwargs)
        self.num_stages = num_stages
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.use_residual = use_residual

        self.tcn_list = []
        for i in range(num_stages + 1):
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
        self.normalization_layers = [keras.layers.LayerNormalization(epsilon=1e-6)]
        for i in range(num_stages - 1):
            self.normalization_layers.append(
                keras.layers.LayerNormalization(epsilon=1e-6)
            )
            self.normalization_layers.append(
                keras.layers.LayerNormalization(epsilon=1e-6)
            )

        self.attention_list = []
        for i in range(num_stages):
            self.attention_list.append(
                MultiHeadAttention(
                    key_dim=self.key_size,
                    value_dim=self.value_size,
                    num_heads=self.num_attention_heads,
                )
            )

    def call(self, inputs):
        data_encoder, data_decoder = inputs
        out_tcn = self.tcn_list[0](data_decoder)
        out = self.attention_list[0]([data_encoder, out_tcn])
        out = self.normalization_layers[0](out_tcn + out)

        for i in range(self.num_stages - 1):
            out_tcn = self.tcn_list[i + 1](out)
            if self.use_residual:
                out = self.normalization_layers[i + 1](out_tcn + out)
            else:
                out = self.normalization_layers[i + 1](out_tcn)
            out_att = self.attention_list[i + 1]([data_encoder, out])
            out = self.normalization_layers[i + 2](out_att + out)
        out = self.tcn_list[-1](out)
        return out