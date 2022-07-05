import tensorflow as tf

from tcn_sequence_models.tf_models.tf_TCN_GRU.decoder import Decoder
from tcn_sequence_models.tf_models.tf_TCN_GRU.encoder import Encoder


class TCN_GRU(tf.keras.Model):
    def __init__(
        self,
        hidden_units: int,
        num_filters: int,
        kernel_size_enc: int,
        dilation_base: int,
        dropout_rate: float,
        gru_output_neurons: [int],
        output_size_attention: int,
        key_size: int,
        value_size: int,
        num_attention_heads: int = 1,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding_encoder: str = "causal",
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        """Model that uses a TCN based encoder and a GRU based decoder

        To get further information about the encoder and decoder architecture,
        read the docstrings of those.

        :param hidden_units: Number of GRU hidden units
        :param num_filters: number of filters / channels used. Also defines the
        number of hidden state units of the decoder GRU
        :param kernel_size_enc: kernel size of the encoder TCN
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param gru_output_neurons: list of output neurons. Each entry is a new layer in
        the output stage.
        :param output_size_attention: output size of the final cross attention. This
        is the number of values that are used as additional inputs for the Decoder GRU
        :param key_size: key size for attention
        :param value_size: value size for attention
        :param num_attention_heads: number of attention heads
        :param activation: activation function
        :param kernel_initializer: kernel initializer
        :param padding_encoder: Padding mode of the encoder. One of ['causal', 'same']
        :param batch_norm: whether to use batch normalization
        :param layer_norm: whether to use layer normalization
        """
        self.hidden_units = hidden_units
        self.num_filters = num_filters
        self.kernel_size_enc = kernel_size_enc
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.gru_output_neurons = gru_output_neurons
        self.output_size_attention = output_size_attention
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding_enc = padding_encoder
        self.batch_norm_tcn = batch_norm
        self.layer_norm_tcn = layer_norm

        super(TCN_GRU, self).__init__()

    def build(self, input_shape):

        self.encoder = Encoder(
            max_seq_len=input_shape[0][1],
            num_filters=self.num_filters,
            kernel_size=self.kernel_size_enc,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding_enc,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
        )

        self.decoder = Decoder(
            units=self.hidden_units,
            output_neurons=self.gru_output_neurons,
            key_size=self.key_size,
            value_size=self.value_size,
            num_attention_heads=self.num_attention_heads,
            output_size_attention=self.output_size_attention,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
        )

    @tf.function
    def call(self, inputs, training=None):
        # if training is False:
        x_encoder, x_decoder, last_y = inputs
        enc_out = self.encoder(x_encoder, training=training)
        predictions = self.decoder([enc_out, x_decoder, last_y], training=training)

        return predictions
