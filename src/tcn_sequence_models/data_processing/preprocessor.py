import json
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import os

import sys
from pathlib import Path

import tcn_sequence_models.utils.train_test_split
from tcn_sequence_models import utils
from tcn_sequence_models.data_processing.preprocessing import OneHotEncoder, NaNHandler

sys.path.insert(0, str(Path().cwd() / Path("../..")) + str(Path("/")))
sys.path.insert(0, str(Path().cwd() / Path("../../..")) + str(Path("/")))

from tcn_sequence_models.data_processing import gen_sequences, preprocessing


class Preprocessor:
    """Preprocessor class to prepare data for the models"""

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df
        self.df_processed = None
        self.features_input_encoder = None
        self.features_input_decoder = None
        self.feature_target = None
        self.input_seq_len = None
        self.output_seq_len = None
        self.temporal_encodings = []
        self.model_type = None
        self.scaler_X = None
        self.scaler_y = None
        self.X = None
        self.y = None
        self.autoregressive = False
        self.nan_handler = None
        self.one_hot_encoder = None
        self.time_col = None
        self.min_rel_occurrence = None

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
        features_input_encoder: List[str],
        features_input_decoder: List[str],
        feature_target: str,
        input_seq_len: int,
        output_seq_len: int,
        model_type: str,
        time_col: Optional[str] = None,
        split_ratio: Optional[float] = None,
        split_date=None,
        temporal_encoding_modes: Optional[List[str]] = None,
        min_rel_occurrence: Optional[float] = None,
        autoregressive: bool = False,
    ):
        """Process the raw data

        This function executed the following steps:
        1. Fill NaNs by using the last observed value in the column
        2. One-hot-encode categorical features
        3. Add temporal encodings as defined in the temporal_encoding_modes parameter
        4. Scale the features using a StandardScaler. The scalers for the input
        features and target feature are saved in the scaler_X and scaler_y
        attributes, respectively.
        5. Create sequences for encoder and decoder inputs and the target values. The
        sequences are stored in the X and y attributes.


        :param features_input_encoder: list of the encoder input features
        :param features_input_decoder: list of the decoder input features
        :param feature_target: label of the target feature
        :param input_seq_len: input (encoder) sequence length
        :param output_seq_len: output (decoder and target) sequence length
        :param model_type: One of ['tcn_tcn', 'tcn_gru']
        :param split_ratio: the ratio with which to split into train and test set
        :param split_date: the date with which to split into train and test set
        :param temporal_encoding_modes: list of the temporal encodings to apply.
        Possible encodings: 'hours', 'months', 'seasons', 'weekdays', 'holidays'
        :param min_rel_occurrence: minimum relative number of occurrences of
        categorical column values to be used for one-hot-encoding.
        :param autoregressive: if True, the X attribute gets the last target value
        as third element. This last element can be used as a first input of a decoder
        that reuses past predictions (e.g. when using an RNN as decoder)
        :return:
        """

        assert split_ratio is not None or split_date is not None, (
            "split_ratio or " "split_date must be " "not None "
        )

        if model_type not in ["tcn_tcn", "tcn_gru"]:
            raise ValueError(
                "model_type must be one of the following: ['tcn_tcn', 'tcn_gru']"
            )

        self.time_col = time_col
        self.df_processed = self.df_raw.copy()
        self.autoregressive = autoregressive
        self.features_input_encoder = features_input_encoder
        self.features_input_decoder = features_input_decoder
        self.feature_target = feature_target
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.model_type = model_type
        self.min_rel_occurrence = min_rel_occurrence

        # compute split ratio if from split_date
        if split_date is not None:
            i_split = self.df_processed[
                self.df_processed["date / time"].dt.date < split_date
            ].index[-1]
            split_ratio = i_split / self.df_processed.shape[0]

        # NaN handling
        self.nan_handler = NaNHandler()
        self.nan_handler.fit(self.df_processed)
        self.nan_handler.transform(self.df_processed, inplace=True)

        # One-hot-encoding
        self.one_hot_encoder = OneHotEncoder(min_rel_occurrence=min_rel_occurrence)
        self.one_hot_encoder.fit(self.df_processed)
        self.one_hot_encoder.transform(df=self.df_processed, inplace=True)

        # Create lists with new features
        encoded_features_input_encoder = self._create_encoded_feature_lists(
            features_input_encoder, self.one_hot_encoder
        )
        encoded_features_input_decoder = self._create_encoded_feature_lists(
            features_input_decoder, self.one_hot_encoder
        )

        # Add temporal encoding
        if temporal_encoding_modes is None:
            temporal_encoding_modes = []
        for temp_enc in temporal_encoding_modes:
            self.df_processed, temporal_encoding = preprocessing.add_temporal_encoding(
                self.df_processed,
                self.time_col,
                split_ratio,
                feature=feature_target,
                mode=temp_enc,
            )
            encoded_features_input_encoder = encoded_features_input_encoder + [
                "temporal_encoding_" + temp_enc
            ]
            encoded_features_input_decoder = encoded_features_input_decoder + [
                "temporal_encoding_" + temp_enc
            ]

            self.temporal_encodings.append((temp_enc, temporal_encoding))

        # scale X-features
        self.df_processed, self.scaler_X = utils.scaling.scale_input_data(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
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
        (
            X_encoder,
            X_decoder,
            y,
            y_shifted,
            y_last,
        ) = gen_sequences.extract_sequences_encoder_decoder_training(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            downsampling_ratio_encoder=1,
            downsampling_ratio_decoder=1,
        )
        if self.model_type == "tcn_tcn":
            if self.autoregressive:
                self.X = [X_encoder, X_decoder, y_shifted, y_last]
            else:
                if X_decoder.shape[-1] == 0:
                    X_decoder = np.full(
                        shape=(X_decoder.shape[0], X_decoder.shape[1], 1),
                        fill_value=1,
                        dtype=float,
                    )
                self.X = [X_encoder, X_decoder]
        else:
            self.X = [X_encoder, X_decoder, y_last]

        self.y = y

    def _create_encoded_feature_lists(
        self, feature_names: List[str], ohe: OneHotEncoder
    ) -> List[str]:
        """Create list with encoded features.

        :param feature_names: List of original feature names
        :param ohe: Fitted OneHotEncoder
        :return: List with encoded feature names
        """
        new_feature_names = []
        for f in feature_names:
            if f in ohe.new_col_names.keys():
                new_feature_names += ohe.new_col_names[f]
            else:
                new_feature_names.append(f)
        return new_feature_names

    def process_from_config_inference(
        self,
        input_seq_len: Optional[int] = None,
        output_seq_len: Optional[int] = None,
    ):
        """Process the raw data from an existing Preprocessor configuration for
        inference.
        Use this method when the preprocessor was loaded from a config and the data will
        be used for inference. The processed data that is used to make predictions is
        stored in self.X. No ground truth values of the predictions are extracted.

        :param input_seq_len: input (encoder) sequence length
        :param output_seq_len: output (decoder and target) sequence length

        :return:
        """
        if input_seq_len is not None:
            self.input_seq_len = input_seq_len
        if output_seq_len is not None:
            self.output_seq_len = output_seq_len

        self.df_processed = self.df_raw.copy()

        # NaN handling
        self.nan_handler.transform(self.df_processed, inplace=True)

        # one-hot-encoding
        self.one_hot_encoder.transform(self.df_processed, inplace=True)

        # Create lists with new features
        encoded_features_input_encoder = self._create_encoded_feature_lists(
            self.features_input_encoder, self.one_hot_encoder
        )
        encoded_features_input_decoder = self._create_encoded_feature_lists(
            self.features_input_decoder, self.one_hot_encoder
        )

        # Add temporal encoding
        for temp_enc in self.temporal_encodings:
            self.df_processed, temporal_encoding = preprocessing.add_temporal_encoding(
                self.df_processed,
                time_col=self.time_col,
                mode=temp_enc[0],
                encoding=temp_enc[1],
            )
            encoded_features_input_encoder.append("temporal_encoding_" + temp_enc[0])
            encoded_features_input_decoder.append("temporal_encoding_" + temp_enc[0])

        # scale X-features
        self.df_processed, _ = utils.scaling.scale_input_data(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
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

        (
            X_encoder,
            X_decoder,
            y_shifted,
            y_last,
        ) = gen_sequences.extract_sequences_encoder_decoder_inference(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
            self.feature_target,
            self.input_seq_len,
            self.output_seq_len,
            downsampling_ratio_encoder=1,
            downsampling_ratio_decoder=1,
        )

        if self.model_type == "tcn_tcn":
            if self.autoregressive:
                self.X = [X_encoder, X_decoder, y_last]
            else:
                self.X = [X_encoder, X_decoder]
        else:
            self.X = [X_encoder, X_decoder, y_last]

    def process_from_config_training(
        self,
        input_seq_len: int,
        output_seq_len: int,
    ):
        """Process the raw data from an existing Preprocessor configuration for
        training.
        Use this method when the preprocessor was loaded from a config and a
        train_test_split is performed afterwards.

        :param input_seq_len: input (encoder) sequence length
        :param output_seq_len: output (decoder and target) sequence length

        :return:
        """
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        # Load config
        self.df_processed = self.df_raw.copy()

        # NaN handling
        self.nan_handler.transform(self.df_processed, inplace=True)

        # one-hot-encoding
        self.one_hot_encoder.transform(self.df_processed, inplace=True)

        # Create lists with new features
        encoded_features_input_encoder = self._create_encoded_feature_lists(
            self.features_input_encoder, self.one_hot_encoder
        )
        encoded_features_input_decoder = self._create_encoded_feature_lists(
            self.features_input_decoder, self.one_hot_encoder
        )

        # Add temporal encoding
        for temp_enc in self.temporal_encodings:
            self.df_processed, temporal_encoding = preprocessing.add_temporal_encoding(
                self.df_processed,
                time_col=self.time_col,
                mode=temp_enc[0],
                encoding=temp_enc[1],
            )
            encoded_features_input_encoder.append("temporal_encoding_" + temp_enc[0])
            encoded_features_input_decoder.append("temporal_encoding_" + temp_enc[0])

        # scale X-features
        self.df_processed, _ = utils.scaling.scale_input_data(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
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

        (
            X_encoder,
            X_decoder,
            y,
            y_shifted,
            y_last,
        ) = gen_sequences.extract_sequences_encoder_decoder_training(
            self.df_processed,
            encoded_features_input_encoder,
            encoded_features_input_decoder,
            self.feature_target,
            self.input_seq_len,
            self.output_seq_len,
            downsampling_ratio_encoder=1,
            downsampling_ratio_decoder=1,
        )

        if self.model_type == "tcn_tcn":
            if self.autoregressive:
                self.X = [X_encoder, X_decoder, y_shifted, y_last]
            else:
                self.X = [X_encoder, X_decoder]
        else:
            self.X = [X_encoder, X_decoder, y_last]
        self.y = y

    def save_preprocessor_config(self, save_path):
        """Save the Preprocessor configuration including the scalers

        :param save_path: directory to save the Preprocessor configuration
        :return:
        """

        # Create and save json of preprocessor configs
        config_file_dir = os.path.join(save_path, "preprocessor_config.json")
        config_dict = {}
        config_dict["features_input_encoder"] = self.features_input_encoder
        config_dict["features_input_decoder"] = self.features_input_decoder
        config_dict["feature_target"] = self.feature_target
        config_dict["temporal_encoding"] = self.temporal_encodings
        config_dict["autoregressive"] = self.autoregressive
        config_dict["input_seq_len"] = self.input_seq_len
        config_dict["output_seq_len"] = self.output_seq_len
        config_dict["model_type"] = self.model_type
        config_dict["time_col"] = self.time_col

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save NaNHandler
        nan_handler_dir = os.path.join(save_path, "NaNHandler.pkl")
        with open(nan_handler_dir, "wb") as f:
            pickle.dump(self.nan_handler, f, pickle.HIGHEST_PROTOCOL)

        # Save OneHotEncoder
        ohe_dir = os.path.join(save_path, "OneHotEncoder.pkl")
        with open(ohe_dir, "wb") as f:
            pickle.dump(self.one_hot_encoder, f, pickle.HIGHEST_PROTOCOL)

        # Save scalers
        scaler_X_dir = os.path.join(save_path, "scaler_X.pkl")
        with open(scaler_X_dir, "wb") as f:
            pickle.dump(self.scaler_X, f, pickle.HIGHEST_PROTOCOL)

        scaler_y_dir = os.path.join(save_path, "scaler_y.pkl")
        with open(scaler_y_dir, "wb") as f:
            pickle.dump(self.scaler_y, f, pickle.HIGHEST_PROTOCOL)

    def load_preprocessor_config(self, load_path):
        """Load a saved Preprocessor configuration

        :param load_path: the directory from where to load the configuration
        :return:
        """

        # Load Preprocessor config
        config_file_dir = os.path.join(load_path, "preprocessor_config.json")
        config_dict = json.load(open(config_file_dir))

        self.features_input_encoder = config_dict["features_input_encoder"]
        self.features_input_decoder = config_dict["features_input_decoder"]
        self.feature_target = config_dict["feature_target"]
        self.temporal_encodings = config_dict["temporal_encoding"]
        self.autoregressive = config_dict["autoregressive"]
        self.input_seq_len = config_dict["input_seq_len"]
        self.output_seq_len = config_dict["output_seq_len"]
        self.model_type = config_dict["model_type"]
        self.time_col = config_dict["time_col"]

        nan_handler_dir = os.path.join(load_path, "NaNHandler.pkl")
        with open(nan_handler_dir, "rb") as f:
            self.nan_handler = pickle.load(f)

        ohe_dir = os.path.join(load_path, "OneHotEncoder.pkl")
        with open(ohe_dir, "rb") as f:
            self.one_hot_encoder = pickle.load(f)
        scaler_X_dir = os.path.join(load_path, "scaler_X.pkl")
        scaler_y_dir = os.path.join(load_path, "scaler_y.pkl")
        with open(scaler_X_dir, "rb") as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_dir, "rb") as f:
            self.scaler_y = pickle.load(f)

    def train_test_split(self, split_ratio):
        """Split data into training and test set using a defined splitting ratio.

        :param split_ratio: ratio between training and test set
        :return: 4 lists containing the input training data, output training data,
        input test data and output test data
        """
        (
            X_train,
            y_train,
            X_val,
            y_val,
        ) = tcn_sequence_models.utils.train_test_split.train_test_split(
            self.X, self.y, split_ratio
        )
        if self.model_type == "tcn_tcn" and self.autoregressive:
            X_train = X_train[:3]
            X_val = X_val[:2] + [X_val[3]]

        return X_train, y_train, X_val, y_val
