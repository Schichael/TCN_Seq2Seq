import tensorflow as tf

from .decoder import Decoder
from .encoder import Encoder


class TCN_Seq2Seq(tf.keras.Model):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        key_size: int,
        value_size: int,
        num_attention_heads: int,
        neurons_output: [int],
        num_layers_tcn: int = None,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm_tcn: bool = False,
        layer_norm_tcn: bool = False,
        padding_encoder: str = "same",
        padding_decoder: str = "causal",
        autoregressive: bool = False,
    ):
        """Model that uses a TCN as encoder and a TCN based decoder.

        To get further information about the encoder and decoder architecture,
        read the docstrings of those.

        :param num_filters: number of filters / channels used. Also defines the
        number of hidden state units of the decoder GRU
        :param kernel_size: kernel size of the TCNs
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param key_size: dimensionality of key/query
        :param value_size: dimensionality of value
        :param num_attention_heads: Number of attention heads
        :param neurons_output: list of output neurons. Each entry is a new layer in
        the output stage.
        :param num_layers_tcn: number of layer in the TCNs. If None, the needed
        number of layers is computed automatically based on the sequence lenghts
        :param activation: the activation function used throughout the model
        :param kernel_initializer: the mode how the kernels are initialized
        :param batch_norm_tcn: if batch normalization shall be used
        :param layer_norm_tcn: if layer normalization shall be used
        :param padding_encoder: Padding mode of the encoder. One of ['causal', 'same']
        :param padding_decoder: Padding mode of the encoder. One of ['causal',
        'same']. If autoregressive = True, decoder padding will always be causal an
        the padding_decoder value has no effect.
        :param autoregressive: whether to use autoregression in the decoder or not.
        If True, teacher-forcing is applied during training and autoregression is
        used during inference. If False, groundtruths / predictions of the previous
        step are not used.
        """
        super(TCN_Seq2Seq, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.neurons_output = neurons_output
        self.num_layers_tcn = num_layers_tcn
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm_tcn = batch_norm_tcn
        self.layer_norm_tcn = layer_norm_tcn
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder
        self.autoregressive = autoregressive

        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        self.encoder = Encoder(
            max_seq_len=input_shape[0][1],
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
            num_layers=self.num_layers_tcn,
            padding=self.padding_encoder,
        )

        self.decoder = Decoder(
            max_seq_len=input_shape[1][1],
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            key_size=self.key_size,
            value_size=self.value_size,
            num_attention_heads=self.num_attention_heads,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
            output_neurons=self.neurons_output,
            num_layers=self.num_layers_tcn,
            autoregressive=self.autoregressive,
            padding=self.padding_decoder,
        )

    @tf.function
    def call(self, inputs, training=None):
        if self.autoregressive:
            x_encoder, x_decoder, y_shifted = inputs
            enc_out = self.encoder(x_encoder, training=training)
            predictions = self.decoder(
                [enc_out, x_decoder, y_shifted], training=training
            )
        else:
            x_encoder, x_decoder = inputs
            enc_out = self.encoder(x_encoder, training=training)
            predictions = self.decoder([enc_out, x_decoder], training=training)

        return predictions
