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
