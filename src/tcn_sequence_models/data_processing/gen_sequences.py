import numpy as np


def extract_sequences_encoder_decoder_training(
    df,
    input_features_encoder: [str],
    input_features_decoder: [str],
    target_feature: str,
    encoder_length: int,
    decoder_length: int,
    downsampling_ratio_encoder: int = 1,
    downsampling_ratio_decoder: int = 1,
):
    """Extract sequences from the dataframe for model input for training

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
    inputs_encoder = df[:-decoder_length][input_features_encoder].values
    inputs_decoder = df[encoder_length:][input_features_decoder].values

    y_shifted = df[(encoder_length - 1) : -1][target_feature].values

    outputs_decoder = df[encoder_length:][target_feature].values

    outputs_encoder_last = df[(encoder_length - 1) : -decoder_length][
        target_feature
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


def extract_sequences_encoder_decoder_inference(
    df,
    input_features_encoder: [str],
    input_features_decoder: [str],
    target_feature: str,
    encoder_length: int,
    decoder_length: int,
    downsampling_ratio_encoder: int = 1,
    downsampling_ratio_decoder: int = 1,
):
    """Extract sequences from the dataframe for model input for inference

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
    inputs_encoder = df[:-decoder_length][input_features_encoder].values
    inputs_decoder = df[encoder_length:][input_features_decoder].values

    y_shifted = df[(encoder_length - 1) : -1][target_feature].values

    outputs_encoder_last = df[(encoder_length - 1) : -decoder_length][
        target_feature
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
    return X_encoder, X_decoder, y_shifted, y_last


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
