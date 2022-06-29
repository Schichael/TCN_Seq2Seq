import numpy as np


def train_test_split(
    sequences_X,
    sequences_y,
    split_ratio: float,
):
    """Splits the sequences into training and test data.

    :param sequences_X: input data sequences
    :param sequences_y: target data sequences
    :param split_ratio: the ratio to split between train and test set
    :return: the training and test data
    """

    assert 0 < split_ratio <= 1.0, "ratio must be between 0 and 1."

    num_sequences = sequences_X[0].shape[0]
    len_sequence = sequences_X[0].shape[1] + sequences_X[1].shape[1]
    X_train = []
    X_val = []
    num_sequences_train = int((num_sequences - len_sequence) * split_ratio)
    num_sequences_test = (num_sequences - len_sequence) - num_sequences_train
    for el in sequences_X:
        X_train.append(el[:num_sequences_train])
        X_val.append(el[-num_sequences_test:])
    y_train = sequences_y[:num_sequences_train]
    y_val = sequences_y[-num_sequences_test:]

    return X_train, y_train, X_val, y_val


def extract_sequences_encoder_decoder(
    df,
    input_features_encoder: [str],
    input_features_decoder: [str],
    target_feature: str,
    encoder_length: int,
    decoder_length: int,
    parent_learning: bool = True,
    downsampling_ratio_encoder: int = 1,
    downsampling_ratio_decoder: int = 1,
):
    """Extract sequences from the dataframe for model input

    :param df: input DataFrame
    :param input_features_encoder: features used for the encoder
    :param input_features_decoder: features used for the decoder
    :param target_feature: label of the target feature
    :param encoder_length: length / number of time steps of the encoder
    :param decoder_length: length / number of time steps of the decoder
    :param downsampling_ratio_encoder: downsampling ratio to use for encoder data
    :param downsampling_ratio_decoder: downsampling ratio to use for decoder data
    :return: arrays of encoder input, decoder input, target values and target values
    of last encoder timestep
    """
    inputs_encoder = df[: -decoder_length][
        input_features_encoder
    ].values
    inputs_decoder = df[encoder_length :][
        input_features_decoder
    ].values

    y_shifted = df[(encoder_length - 1) :-1][target_feature].values

    outputs_decoder = df[encoder_length :][
        target_feature
    ].values

    outputs_encoder_last = df[
        (encoder_length - 1) : -decoder_length
    ][target_feature].values

    X_encoder = _extract_windows(
        array=inputs_encoder,
        seq_length=encoder_length,
        downsampling_ratio=downsampling_ratio_encoder,
    )
    X_decoder = _extract_windows(
        array=inputs_decoder,
        seq_length=decoder_length,
        downsampling_ratio=downsampling_ratio_decoder,
    )
    y = _extract_windows(
        array=outputs_decoder,
        seq_length=decoder_length,
        downsampling_ratio=downsampling_ratio_decoder,
    )
    y_shifted = _extract_windows(
        array=y_shifted,
        seq_length=decoder_length,
        downsampling_ratio=downsampling_ratio_decoder,
    )

    y_last = _extract_windows(
        array=outputs_encoder_last,
        seq_length=1,
        downsampling_ratio=downsampling_ratio_decoder,
    )
    return X_encoder, X_decoder, y, y_shifted, y_last


def extract_sequences_one_classification_regression(
    df_X,
    df_y,
    input_features_encoder: [str],
    encoder_length: int,
    downsampling_ratio_encoder: int = 1,
):
    """Extract sequences from the dataframe for model input

    :param df: input DataFrame
    :param input_features_encoder: features used for the encoder
    :param target_feature: label of the target feature
    :param encoder_length: length / number of time steps of the encoder
    :param downsampling_ratio_encoder: downsampling ratio to use for encoder data
    :return: arrays of encoder input, decoder input, target values and target values
    of last encoder timestep
    """
    inputs_encoder = df_X[
        input_features_encoder
    ].values

    outputs = df_y[
        encoder_length:
    ].values

    X_encoder = _extract_windows(
        array=inputs_encoder,
        seq_length=encoder_length,
        downsampling_ratio=downsampling_ratio_encoder,
    )

    y = _extract_windows(
        array=outputs,
        seq_length=1,
        downsampling_ratio=1
    )

    return X_encoder, y


def extract_sequences_inference_encoder_decoder(
    df,
    input_features_encoder: [str],
    input_features_decoder: [str],
    target_feature: str,
    encoder_length: int,
    decoder_length: int,
    downsampling_ratio_encoder: int = 1,
    downsampling_ratio_decoder: int = 1,
    autoregressive: bool = False
):
    """Extract sequences from the dataframe for model input for inference,
    i.e. without the target features.

    :param df: input DataFrame
    :param input_features_encoder: features used for the encoder
    :param input_features_decoder: features used for the decoder
    :param target_feature: label of the target feature
    :param encoder_length: length / number of time steps of the encoder
    :param decoder_length: length / number of time steps of the decoder
    :param downsampling_ratio_encoder: downsampling ratio to use for encoder data
    :param downsampling_ratio_decoder: downsampling ratio to use for decoder data
    :param autoregressive: whether to have the last y value of the encoder as first
    decoder inout.
    :return: arrays of encoder input, decoder input, target values and target values
    of last encoder timestep
    """
    if len(input_features_decoder) == 0:
        inputs_encoder = df[input_features_encoder].values
        inputs_decoder = df[
            []
        ].values
    else:
        inputs_encoder = df[: -decoder_length * downsampling_ratio_decoder][
            input_features_encoder
        ].values
        inputs_decoder = df[encoder_length * downsampling_ratio_encoder:][
            input_features_decoder
        ].values



    X_encoder = _extract_windows(
        array=inputs_encoder,
        seq_length=encoder_length,
        downsampling_ratio=downsampling_ratio_encoder,
    )
    X_decoder = _extract_windows(
        array=inputs_decoder,
        seq_length=decoder_length,
        downsampling_ratio=downsampling_ratio_decoder,
    )

    outputs_encoder_last = df[
                           (encoder_length - 1)
                           * downsampling_ratio_encoder: -decoder_length
                                                         * downsampling_ratio_decoder
                           ][target_feature].values

    y_last = _extract_windows(
        array=outputs_encoder_last,
        seq_length=1,
        downsampling_ratio=downsampling_ratio_decoder,
    )
    return X_encoder, X_decoder, y_last




def _extract_windows(array, seq_length, downsampling_ratio):
    """Transforms input array into sequences using the sliding window technique

    :param array: input array
    :param seq_length: the number of datapoints for each sequence
    :param downsampling_ratio: the ratio for which the sequences are downsampled.
    E.g. if downsampling_ratio = 2 every second datapoint will be used
    :return: the sequences in form of array of arrays
    """

    assert (
        isinstance(downsampling_ratio, int) or downsampling_ratio == 0
    ), "downsampling_ratio must be an integer greater than 0"
    len_array = array.shape[0]
    window_size_prime = 1 + (seq_length - 1) * downsampling_ratio
    K_indices = np.arange(0, seq_length * downsampling_ratio, step=downsampling_ratio)
    T_indices = np.arange(0, (len_array - window_size_prime + 1), step=1)
    sub_windows = np.round(
        np.expand_dims(K_indices, 0) + np.expand_dims(T_indices, 0).T
    ).astype(int)
    return array[sub_windows]
