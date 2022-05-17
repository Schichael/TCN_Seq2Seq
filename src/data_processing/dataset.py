import json
import pickle

import pandas as pd
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path().cwd() / Path("..")) + str(Path("/")))
sys.path.insert(0, str(Path().cwd() / Path("..//..")) + str(Path("/")))
try:
    from src.data_processing import preprocessing, gen_sequences
except:
    from . import preprocessing, gen_sequences

from src import utils


class DataSet:
    def __init__(self):
        """Dataset class to handle loading and preparing data for the models

        Steps to load and process a dataset to make it ready for the models
        with / without a saved configuration:
        1. call load_data(...)
        2. call process(...) / process_from_config(...)
        3. call train_test_split(...)
        4. to save the dataset configurations: call save_dataset_config(...)

        """

        self.df_raw = None
        self.df_processed = None
        self.features_input_encoder = None
        self.features_input_decoder = None
        self.feature_target = None
        self.input_seq_len = None
        self.output_seq_len = None
        self.temporal_encodings = []
        self.smoothing_operations = None
        self.scaler_X = None
        self.scaler_y = None
        self.X = None
        self.y = None
        self.autoregressive = False

    def load_data(self, path, file_type="xlsx"):
        """Load the raw data from a xlsx or csv file

        The data will be loaded into the df_raw DataFrame attribute

        Note: Currently, only works with xlsx files because of issues with the
        date-time data when using csv files.

        :param path: the path to the data file
        :param file_type: the type of the file. Currently only 'xlsx' is supported
        :return:
        """
        if file_type == "xlsx":
            self.df_raw = pd.read_excel(path)
        elif file_type == "csv":
            raise NotImplementedError("Currently only xlsx files are supported")
        else:
            raise ValueError("Currently only xlsx files are supported")

    def process(
        self,
        features_input_encoder: [str],
        features_input_decoder: [str],
        feature_target: str,
        input_seq_len: int,
        output_seq_len: int,
        split_ratio: float = None,
        split_date=None,
        smoothing_operations=None,
        temporal_encoding_modes: [str] = None,
        autoregressiive: bool = False,
    ):
        """Process the raw data

        This function executed the following steps:
        1. Remove days from the dataset where MeteoViva is inactive
        2. Fill NaNs by using the last observed value in the column
        3. Add temporal encodings as defined in the temporal_encoding_modes parameter
        4. Perform smoothing as defined in the smoothing_operations parameter
        5. Scale the features that are defined in features_input_encoder,
        features_input_decoder and feature_target using a StandardScaler. The
        scalers for the input features and target feature are saved in the scaler_X
        and scaler_y attributes, respectively.
        6. Create sequences for encoder and decoder inputs and the target values. The
        sequences are stored in the X and y attributes.


        :param features_input_encoder: list of the encoder input features
        :param features_input_decoder: list of the decoder input features
        :param feature_target: label of the target feature
        :param input_seq_len: input (encoder) sequence length
        :param output_seq_len: output (decoder and target) sequence length
        :param split_ratio: the ratio with which to split into train and test set
        :param split_date: the date with which to split into train and test set
        :param smoothing_operations: list of tuples that define the smoothing
        operations of the form: [(method, window_size, features)]. method can be
        'median' or 'mean'. A rolling window is used for smoothing
        :param temporal_encoding_modes: list of the temporal encodings to apply.
        Possible encodings: 'hours', 'months', 'seasons', 'weekdays', 'holidays'
        :param autoregressiive: if True, the X attribute gets the last target value
        as third element. This last element can be used as a first input of a decoder
        that reuses past predictions (e.g. when using an RNN as decoder)
        :return:
        """

        assert split_ratio is not None or split_date is not None, (
            "split_ratio or " "split_date must be " "not None "
        )

        self.autoregressive = autoregressiive
        self.features_input_encoder = features_input_encoder
        self.features_input_decoder = features_input_decoder
        self.feature_target = feature_target
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # Remove inactive days
        self.df_processed = preprocessing.remove_inactive_days(self.df_raw)

        # Fill gaps
        self.df_processed = preprocessing.fill_gaps(self.df_processed)

        # compute split ratio if from split_date
        if split_date is not None:
            i_split = self.df_processed[
                self.df_processed["date / time"].dt.date < split_date
            ].index[-1]
            split_ratio = i_split / self.df_processed.shape[0]

        # Add temporal encoding
        if temporal_encoding_modes is None:
            temporal_encoding_modes = []
        for temp_enc in temporal_encoding_modes:
            print(temp_enc)
            self.df_processed, temporal_encoding = preprocessing.add_temporal_encoding(
                self.df_processed,
                split_ratio,
                feature=feature_target,
                mode=temp_enc,
            )
            features_input_encoder = features_input_encoder + [
                "temporal_encoding_" + temp_enc
            ]
            features_input_decoder = features_input_decoder + [
                "temporal_encoding_" + temp_enc
            ]

            self.temporal_encodings.append((temp_enc, temporal_encoding))

        # Add smoothing
        if smoothing_operations is not None:
            for smoothing_op in smoothing_operations:
                self.df_processed = preprocessing.smooth_rolling_window(
                    self.df_processed,
                    features=smoothing_op[2],
                    window_size=smoothing_op[1],
                    method=smoothing_op[0],
                    delete_nans=True,
                )
        self.smoothing_operations = smoothing_operations

        # scale X-features
        self.df_processed, self.scaler_X = utils.scaling.scale_input_data(
            self.df_processed,
            features_input_encoder,
            features_input_decoder,
            feature_target,
            train_ratio=split_ratio,
        )

        # scale target
        self.df_processed, self.scaler_y = utils.scaling.scale_target_data(
            self.df_processed, [feature_target], train_ratio=split_ratio
        )

        # Reset index
        self.df_processed = self.df_processed.reset_index(drop=True)

        # Create sequences
        (X_encoder, X_decoder, y, y_last) = gen_sequences.extract_sequences(
            self.df_processed,
            features_input_encoder,
            features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            downsampling_ratio_encoder=1,
            downsampling_ratio_decoder=1,
        )

        if autoregressiive:
            self.X = [X_encoder, X_decoder, y_last]
        else:
            self.X = [X_encoder, X_decoder]

        self.y = y

    def process_from_config(
        self,
        config_path: str,
        input_seq_len: int = None,
        output_seq_len: int = None,
    ):
        """Process the raw data from a an existing DataSet configuration

        This function executed the following steps:
        1. Remove days from the dataset where MeteoViva is inactive
        2. Fill NaNs by using the last observed value in the column
        3. Add temporal encodings as defined in the temporal_encoding_modes parameter
        4. Perform smoothing as defined in the smoothing_operations parameter
        5. Scale the features that are defined in features_input_encoder,
        features_input_decoder and feature_target using a StandardScaler. The
        scalers for the input features and target feature are saved in the scaler_X
        and scaler_y attributes, respectively.
        6. Create sequences for encoder and decoder inputs and the target values. The
        sequences are stored in the X and y attributes.

        :param config_path: the path to the folder in which the config files are stored
        :param input_seq_len: input (encoder) sequence length
        :param output_seq_len: output (decoder and target) sequence length

        :return:
        """
        if input_seq_len is not None:
            self.input_seq_len = input_seq_len
        if output_seq_len is not None:
            self.output_seq_len = output_seq_len
        # Load config
        self.load_dataset_config(config_path)

        # Remove inactive days
        self.df_processed = preprocessing.remove_inactive_days(self.df_raw)

        # Fill gaps
        self.df_processed = preprocessing.fill_gaps(self.df_processed)

        # Add temporal encoding

        for temp_enc in self.temporal_encodings:
            self.df_processed, temporal_encoding = preprocessing.add_temporal_encoding(
                self.df_processed,
                mode=temp_enc[0],
                encoding=temp_enc[1],
            )
            self.features_input_encoder.append("temporal_encoding_" + temp_enc[0])
            self.features_input_decoder.append("temporal_encoding_" + temp_enc[0])

        # Add smoothing
        if self.smoothing_operations:
            for smoothing_op in self.smoothing_operations:
                self.df_processed = preprocessing.smooth_rolling_window(
                    self.df_processed,
                    features=smoothing_op[2],
                    window_size=smoothing_op[1],
                    method=smoothing_op[0],
                    delete_nans=True,
                )

        # scale X-features
        self.df_processed, _ = utils.scaling.scale_input_data(
            self.df_processed,
            self.features_input_encoder,
            self.features_input_decoder,
            self.feature_target,
            scaler=self.scaler_X,
        )

        # scale target
        self.df_processed, _ = utils.scaling.scale_target_data(
            self.df_processed,
            [self.feature_target],
            scaler=self.scaler_y,
        )

        # Reset index
        self.df_processed = self.df_processed.reset_index(drop=True)

        (X_encoder, X_decoder, y, y_last) = gen_sequences.extract_sequences(
            self.df_processed,
            self.features_input_encoder,
            self.features_input_decoder,
            self.feature_target,
            self.input_seq_len,
            self.output_seq_len,
            downsampling_ratio_encoder=1,
            downsampling_ratio_decoder=1,
        )

        if self.autoregressive:
            self.X = [X_encoder, X_decoder, y_last]
        else:
            self.X = [X_encoder, X_decoder]

        self.y = y

    def save_dataset_config(self, save_path):
        """Save the DataSet configuration including the scalers

        :param save_path: directory to save the DataSet configuration
        :return:
        """

        # Create and save json of dataset configs
        config_file_dir = os.path.join(save_path, "dataset_config.json")
        config_dict = {}
        config_dict["features_input_encoder"] = self.features_input_encoder
        config_dict["features_input_decoder"] = self.features_input_decoder
        config_dict["feature_target"] = self.feature_target
        config_dict["temporal_encoding"] = self.temporal_encodings
        config_dict["smoothing_operations"] = self.smoothing_operations
        config_dict["autoregressive"] = self.autoregressive
        config_dict["input_seq_len"] = self.input_seq_len
        config_dict["output_seq_len"] = self.output_seq_len

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save scalers

        scaler_X_dir = os.path.join(save_path, "scaler_X.pkl")
        with open(scaler_X_dir, "wb") as f:
            pickle.dump(self.scaler_X, f, pickle.HIGHEST_PROTOCOL)

        scaler_y_dir = os.path.join(save_path, "scaler_y.pkl")
        with open(scaler_y_dir, "wb") as f:
            pickle.dump(self.scaler_y, f, pickle.HIGHEST_PROTOCOL)

    def load_dataset_config(self, load_path):
        """Load a saved DataSet configuration

        :param load_path: the directory from where to load the configuration
        :return:
        """

        # Load Dataset config
        config_file_dir = os.path.join(load_path, "dataset_config.json")
        config_dict = json.load(open(config_file_dir))

        self.features_input_encoder = config_dict["features_input_encoder"]
        self.features_input_decoder = config_dict["features_input_decoder"]
        self.feature_target = config_dict["feature_target"]
        self.temporal_encodings = config_dict["temporal_encoding"]
        self.smoothing_operations = config_dict["smoothing_operations"]
        self.autoregressive = config_dict["autoregressive"]
        self.input_seq_len = config_dict["input_seq_len"]
        self.output_seq_len = config_dict["output_seq_len"]

        scaler_X_dir = os.path.join(load_path, "scaler_X.pkl")
        scaler_y_dir = os.path.join(load_path, "scaler_y.pkl")
        with open(scaler_X_dir, "rb") as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_dir, "rb") as f:
            self.scaler_y = pickle.load(f)

    def create_dataframe_with_predictions(
        self, predictions, date_time_column="date / time"
    ):
        """Creates a DataFrame with the date and time, true target value and
        predictions for the next timesteps defined in self.output_seq_len

        :param predictions: numpy array with the predictions. It is assumed that it
        includes the predictions for the whole dataset
        :param date_time_column: the label of the date_time column
        :return: DataFrame with 'date / time', target ground truth and for each row
        the next predictions as an array
        """
        truth_scaled = self.df_processed[self.feature_target]

        df = self.df_processed[date_time_column].to_frame()

        # inverse scale
        truth = self.scaler_y.inverse_transform(
            truth_scaled.values.reshape(-1, 1)
        ).reshape(-1)
        df["target"] = truth.tolist()

        predictions_list = predictions.tolist()
        empty_front = [[]] * (self.input_seq_len - 1)
        empty_end = [[]] * self.output_seq_len
        predictions_list = empty_front + predictions_list + empty_end

        df["predictions"] = predictions_list
        return df

    def train_test_split(self, split_ratio):
        """Split data into training and test set using a defined splitting ratio.

        :param split_ratio: ratio between training and test set
        :return: 4 lists containing the input training data, output training data,
        input test data and output test data
        """
        return gen_sequences.train_test_split(self.X, self.y, split_ratio)

    def train_test_split_date(self, split_date, considered_months=None):
        """Split data into training and test set using a date on which to split.

        :param split_date: the date on which the data is split
        :param considered_months: optional list that contains the months that shall
        be included. If None, all months will be used.
        :return: 4 lists containing the input training data, output training data,
        input test data and output test data
        """
        if considered_months is None:
            considered_months = list(range(1, 13))
        i_split = self.df_processed[
            self.df_processed["date / time"].dt.date < split_date
        ].index[-1]
        split_ratio = i_split / self.df_processed.shape[0]

        X_train, y_train, X_test, y_test = gen_sequences.train_test_split(
            self.X, self.y, split_ratio
        )

        i_months = self.df_processed[
            self.df_processed["date / time"].dt.month.isin(considered_months)
        ].index

        # remove indexes that are too large
        i_months = i_months[i_months < X_train[0].shape[0] + X_test[0].shape[0]]

        i_train = i_months[i_months < X_train[0].shape[0]]
        i_test = i_months[i_months >= i_train[-1]] - X_train[0].shape[0]

        X_train_months = []
        y_train_months = y_train[i_train]
        X_test_months = []
        y_test_months = y_test[i_test]
        for x in X_train:
            X_train_months.append(x[i_train])
        for x in X_test:
            X_test_months.append(x[i_test])

        return X_train_months, y_train_months, X_test_months, y_test_months

        # return gen_sequences.train_test_split(self.X, self.y, split_ratio)
