import tensorflow as tf

from tensorflow.keras.layers import MultiHeadAttention


class Decoder(tf.keras.Model):
    def __init__(
        self,
        units: int,
        output_neurons: [int],
        key_size: int,
        value_size: int,
        num_attention_heads: int,
        output_size_attention: int,
        dropout_rate: float,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
    ):
        """Decoder architecture

        The Decoder architecture is as follows:
        The heart of the decoder is a GRU. The initial state of the GRU is a vector
        which values are trained during training.
        The input of the GRU at timestep t consists of the following:
        - The decoder data as defined by the user for timestep t.
        - The last prediction of timestep t-1 or if it's the first timestep the last
        true target value.
        To make a prediction, the current hidden state is concatenated with the last
        output of the cross attention and then fed into a prediction stage (a block
        of dense layers) that then makes the final prediction.

        :param units: number of hidden units
        :param output_neurons: list of output neurons. Each entry is a new layer in
        the output stage.
        :param key_size: key size for attention
        :param value_size: value size for attention
        :param num_attention_heads: number of attention heads
        :param output_size_attention: output size of the final cross attention. This
        is the number of values that are used as additional inputs for the GRU
        :param dropout_rate: dropout rate
        :param batch_norm: whether to use batch normalization
        :param layer_norm: whether to use layer normalization
        :param activation: activation function
        :param kernel_initializer: kernel initializer
        """
        super(Decoder, self).__init__()
        self.units = units
        self.output_neurons = output_neurons
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.output_layers = []
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.output_size_attention = output_size_attention
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        # Initial hidden state for GRU
        self.initial_state = self.add_weight(
            name="initial_state",
            shape=(1, self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        # Classification stage
        for i, neurons in enumerate(self.output_neurons):
            layer_dense = tf.keras.layers.Dense(
                neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
            )
            self.output_layers.append(layer_dense)
        # Final output layer
        self.output_layers.append(tf.keras.layers.Dense(1))

        self.cross_attention = MultiHeadAttention(
            key_dim=self.key_size,
            value_dim=self.value_size,
            num_heads=self.num_attention_heads,
            output_shape=self.output_size_attention,
        )

        self.self_attention = MultiHeadAttention(
            key_dim=self.key_size,
            value_dim=self.value_size,
            num_heads=self.num_attention_heads,
            output_shape=self.output_size_attention,
            # output_dim=self.value_size,
        )

        # Normalization layer after cross attention
        if self.layer_norm:
            self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        else:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs, training=None):
        # predictions variable will be used to store the prediction of each time step
        predictions = None

        enc_out, data_dec, last_y = inputs

        # Put the initial state into the correct form for the GRU and initialize the
        # hidden_state tensor that will be used for self attentionlayer_norm
        initial_state_expanded = tf.expand_dims(self.initial_state, 1)
        hidden_states = tf.tile(initial_state_expanded, [tf.shape(enc_out)[0], 1, 1])
        target_len = data_dec.shape[1]
        dec_hidden = tf.reshape(hidden_states[:, -1, :], [-1, self.units])

        # GRU input without the output of the cross attention
        gru_input = data_dec[:, :1, :]
        last_y = tf.reshape(last_y, [-1, 1, 1])
        gru_input = tf.concat([gru_input, last_y], 2)

        for i in range(target_len):
            # Add output of cross attention to the GRU inputs
            _, dec_hidden = self.gru(
                gru_input, initial_state=dec_hidden, training=training
            )
            # Add current hidden state to the hidden_states tensor
            dec_hidden_reshaped = tf.expand_dims(dec_hidden, 1)
            hidden_states = tf.concat([hidden_states, dec_hidden_reshaped], 1)

            # Apply attentions
            out_self_attention = self.self_attention(
                hidden_states[:, -1:, :], hidden_states, training=training
            )
            out_cross_attention = self.cross_attention(
                out_self_attention, enc_out, training=training
            )
            out_cross_attention = self.normalization_layer(
                out_cross_attention + out_self_attention, training=training
            )
            # output layers
            out = tf.concat(
                [
                    dec_hidden,
                    tf.reshape(out_cross_attention, [-1, self.output_size_attention]),
                ],
                -1,
            )
            for layer in self.output_layers:
                out = layer(out, training=training)

            # Add prediction to the prediction tensor
            if predictions is None:
                predictions = out
            else:
                predictions = tf.concat([predictions, out], 1)
            # Stop loop when the end is reached
            if i == target_len - 1:
                break

            # Define GRU input for next time step
            gru_input = data_dec[:, i + 1 : i + 2, :]
            prediction = tf.expand_dims(out, 1)
            gru_input = tf.concat([gru_input, prediction], 2)

        return predictions
