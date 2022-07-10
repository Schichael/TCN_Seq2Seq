import numpy as np
from sklearn.preprocessing import StandardScaler


def scale(df, columns, train_ratio=1.0, scaler=None):
    """scale the value in a DataFrame

    :param df: the DataFrame
    :param columns: the columns to scale
    :param train_ratio: the ratio of the training data that is used to fit the
    scaler. Leave as 1 if the whole data shall be used to scale
    :param scaler: an already fitted scaler
    :return: the DataFrame with scaled values
    :rtype pandas DataFrame and the scaler
    """
    df = df.copy(deep=True)
    df_train = df[: int(df.shape[0] * train_ratio)]
    if len(columns) == 1:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df_train[columns].values.reshape(-1, 1))
        df[columns] = scaler.transform(df[columns].values.reshape(-1, 1))
    else:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df_train[columns])
        df[columns] = scaler.transform(df[columns])

    return df, scaler


def inverse_scale_sequences(sequences, scaler):
    """inverse scale sequences of data

    :param sequences: input sequences
    :param scaler: fitted scaler
    :return: array of the inverse scaled sequences
    :rtype: numpy array
    """
    unscaled_sequences = []
    for sequence in sequences:
        unscaled_sequences.append(scaler.inverse_transform(sequence))
    return np.array(unscaled_sequences)


def scale_input_data(
    df,
    features_input_encoder,
    features_input_decoder,
    feature_target,
    train_ratio=1.0,
    scaler=None,
):
    """Scales the input data of encoder and decoder

    :param df: DataFrame
    :param features_input_encoder: list of the encoder input features
    :param features_input_decoder: list of the decoder input features
    :param feature_target: label of the target feature
    :param train_ratio: ratio of training data
    :param scaler: optionally a fitted scaler
    :return: DataFrame with scaled data and the fitted scaler
    """
    df = df.copy(deep=True)
    features_input_without_target = list(
        np.unique(features_input_encoder + features_input_decoder)
    )

    # Make sure that target feature doesn't get scaled here
    while feature_target in features_input_without_target:
        features_input_without_target.remove(feature_target)

    df_scaled, scaler = scale(
        df, features_input_without_target, train_ratio=train_ratio, scaler=scaler
    )
    return df_scaled, scaler


def scale_target_data(df, feature_target, train_ratio=1.0, scaler=None):
    """Scales the target data

    :param df: DataFrame
    :param feature_target: label of the target feature
    :param train_ratio: ratio of training data
    :param scaler: optionally a fitted scaler
    :return: DataFrame with scaled data and the fitted scaler
    """
    df = df.copy(deep=True)
    df_scaled, scaler = scale(
        df, feature_target, train_ratio=train_ratio, scaler=scaler
    )
    return df_scaled, scaler
