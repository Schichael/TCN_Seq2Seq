import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadAttention(tf.keras.Model):
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int = 2,
    ):
        """Perform multi-headed attention. This works slightly different as in the
        original paper in the sense that query, key and value are not split before
        feeding them to the heads.

        :param key_dim: dimension of key/query
        :param value_dim: dimension of value
        :param num_heads: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.dense_out = None
        self.key_matrix = None
        self.query_matrix = None
        self.value_matrix = None

    def build(self, input_shape):
        self.key_matrix = self.add_weight(
            name="key_matrix",
            shape=(input_shape[0][-1], self.key_dim * self.num_heads),
            initializer="random_normal",
            trainable=True,
        )

        self.query_matrix = self.add_weight(
            name="query_matrix",
            shape=(input_shape[1][-1], self.key_dim * self.num_heads),
            initializer="random_normal",
            trainable=True,
        )

        self.value_matrix = self.add_weight(
            name="value_matrix",
            shape=(input_shape[0][-1], self.value_dim * self.num_heads),
            initializer="random_normal",
            trainable=True,
        )

        self.dense_out = Dense(input_shape[1][-1])

    def call(self, input):
        encoder_data, decoder_data = input
        key = tf.matmul(encoder_data, self.key_matrix)
        value = tf.matmul(encoder_data, self.value_matrix)
        query = tf.matmul(decoder_data, self.query_matrix)

        # Split key, value and query for multi headed attention
        keys = []
        values = []
        queries = []
        for i in range(self.num_heads):
            keys.append(key[:, :, i * self.key_dim : (i + 1) * self.key_dim])
            values.append(value[:, :, i * self.value_dim : (i + 1) * self.value_dim])
            queries.append(query[:, :, i * self.key_dim : (i + 1) * self.key_dim])
        heads_outputs = []
        for i in range(self.num_heads):
            heads_outputs.append(self._calc_attention(queries[i], values[i], keys[i]))

        heads_outputs_concat = tf.concat(heads_outputs, 2)
        out = self.dense_out(heads_outputs_concat)
        return out

    def _calc_attention(self, query, value, key):
        scores = tf.matmul(query, key, transpose_b=True)
        return tf.linalg.matmul(tf.nn.softmax(scores / key.shape[-1], axis=1), value)
