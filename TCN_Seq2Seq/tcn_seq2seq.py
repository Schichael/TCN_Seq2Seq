import tensorflow as tf

from components.decoder import Decoder
from components.encoder import Encoder


class TCN_Seq2Seq(tf.keras.Model):
    def __init__(
        self,
        num_filters_enc: int,
        num_filters_dec: int,
        kernel_size_enc: int,
        kernel_size_dec: int,
        dilation_base: int,
        dropout_rate: float,
        key_size: int,
        value_size: int,
        num_attention_heads: int,
        neurons_output: [int],
        num_stages_enc: int = 1,
        num_stages_dec: int = 1,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding_enc: str = "causal",
        padding_dec: str = "causal",
        weight_norm_tcn: bool = False,
        batch_norm_tcn: bool = False,
        layer_norm_tcn: bool = False,
        use_residual: bool = True,
    ):
        """Sequence to sequence TCN-based model for time series forecasting.
        Influenced by the Transformer model, it used Multi-Headed
        Encoder-Decoder-attention to connect encoder and decoder. Instead of
        Self-attention as used in the Transformer architecture, this model uses
        TCN stages. Additional positional encoding  is not necessary since a TCN
        stage performs temporal encoding implicitly. Another difference to the
        Transformer is that multi headed attention is computed differently. Here,
        the inputs(query, key, value) are not split before feeding the attention
        heads. This is done due to the usually low dimensionality of the inputs.
        Also, this model does not use auto-correlation in the sense that the t-1th
        prediction is not fed as input to compute the t-th prediction.
        Inputs: [encoder_input, decoder_input]

        :param num_filters_enc: number of channels for encoder-CNNs
        :param num_filters_dec: number of channels for decoder-CNNs
        :param kernel_size_enc: kernel size of encoder-CNNs
        :param kernel_size_dec: kernel size of decoder-CNNs
        :param dilation_base: dilitation base
        :param dropout_rate: dropout rate
        :param key_size: dimensionality of key/query
        :param value_size: dimensionality of value
        :param num_attention_heads: Number of attention heads to be used
        :param neurons_output: list of neurons stacked after the Decoder to compute
        the prediction
        :param num_stages_enc: number of encoder stages stacked on top of each other
        :param num_stages_dec: number of deccoder stages stacked on top of each other
        :param activation: the activation function used throughout the model
        :param kernel_initializer: the mode how the kernels are initialized
        :param padding_enc: padding for encoder, usually' causal' or 'same'
        :param padding_dec: adding for decoder, usually' causal' or 'same'
        :param weight_norm_tcn: if weight normalization shall be used
        :param batch_norm_tcn: if batch normalization shall be used
        :param layer_norm_tcn: if layer normalization shall be used
        :param use_residual: if residual connections shall be used in encoder and
        decoder stacks. (Note: In the TCN-layer, residual connections are always used)
        :param kwargs:
        """
        super(TCN_Seq2Seq, self).__init__()
        self.num_filters_enc = num_filters_enc
        self.num_filters_dec = num_filters_dec
        self.kernel_size_enc = kernel_size_enc
        self.kernel_size_dec = kernel_size_dec
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.neurons_output = neurons_output
        self.num_stages_enc = num_stages_enc
        self.num_stages_dec = num_stages_dec
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding_enc = padding_enc
        self.padding_dec = padding_dec
        self.weight_norm_tcn = weight_norm_tcn
        self.batch_norm_tcn = batch_norm_tcn
        self.layer_norm_tcn = layer_norm_tcn
        self.use_residual = use_residual

        self.encoder = Encoder(
            is_first_block=True,
            num_stages=self.num_stages_enc,
            num_filters=self.num_filters_enc,
            kernel_size=self.kernel_size_enc,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding_enc,
            weight_norm=self.weight_norm_tcn,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
            use_residual=self.use_residual,
        )

        self.decoder = Decoder(
            is_first_block=True,
            num_stages=self.num_stages_dec,
            num_filters=self.num_filters_dec,
            kernel_size=self.kernel_size_dec,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            key_size=self.key_size,
            value_size=self.value_size,
            num_attention_heads=self.num_attention_heads,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding_dec,
            weight_norm=self.weight_norm_tcn,
            batch_norm=self.batch_norm_tcn,
            layer_norm=self.layer_norm_tcn,
            use_residual=self.use_residual,
        )

        self.output_layers = []

        for i, neurons in enumerate(self.neurons_output):
            layer = tf.keras.layers.Dense(
                neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
            )
            self.output_layers.append(layer)

        # last output layer
        layer = tf.keras.layers.Dense(1)
        self.output_layers.append(layer)

    def call(self, inputs):
        x_encoder, x_decoder = inputs
        enc_out = self.encoder(x_encoder)
        out = self.decoder([enc_out, x_decoder])
        for layer in self.output_layers:
            out = layer(out)
        return out
