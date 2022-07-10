from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class NaNHandler:
    """Class to remove / fill NaN values
    For numeric columns, NaNs are replaced by linear interpolation.
    For categorical columns, nothing is done since this is handled by one-hot-encoding.
    If the y-value is categorical, always the last value that was not NaN, will be used.
    If a numerical column was fitted with enough non-NaN values, but when
    transforming, all values are NaN, the missing values are replaced by the median
    value that was computed during fitting.
    """

    def __init__(
        self,
    ):
        self.median_vals = {}

    def fit(self, df: pd.DataFrame):
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in numerical_cols:
            self.median_vals[col] = df[col].median()
            num_nans = df[col].isna().sum()
            if num_nans == 0:
                continue

    def transform(self, df: pd.DataFrame, inplace: bool = False):
        if not inplace:
            df = df.copy()
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Interpolate
        for col in numerical_cols:
            df[col] = df[col].interpolate()
            df[col] = df[col].fillna(method="backfill")
        return df


class OneHotEncoder:
    """
    Class to perform one-hot-encoding. This is done only for categorical data.
    Numerical data will not be one-hot-encoded.
    """

    def __init__(self, min_rel_occurrence: Optional[float] = None):
        self.min_rel_occurrence = min_rel_occurrence
        self.valid_values = {}
        self.new_col_names = {}

    def fit(self, df: pd.DataFrame):
        # Get categorical columns
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols = [column for column in df.columns if is_datetime(df[column])]
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        # Select the values that have enough occurrences for each categorical column
        for col in categorical_cols:
            self.new_col_names[col] = []
            if self.min_rel_occurrence is None:
                unique_vals = df[col].dropna().unique()
                self.valid_values[col] = unique_vals
                self.new_col_names[col] = self._gen_col_name(col, unique_vals)
            else:
                rel_occurrences = df[col].dropna().value_counts(normalize=True)
                high_occurrences = rel_occurrences[
                    rel_occurrences >= self.min_rel_occurrence
                ].index.tolist()
                self.valid_values[col] = high_occurrences
                self.new_col_names[col] = self._gen_col_name(col, high_occurrences)

    def transform(
        self, df: pd.DataFrame, inplace: bool = False, drop_unseen: bool = True
    ):
        if not inplace:
            df = df.copy()
        for col_name, values in self.valid_values.items():
            # Do not transform columns that haven't been seen during fitting. If
            # drop_unseen, drop them.

            if col_name not in df.columns:
                if drop_unseen:
                    df.drop(col_name, inplace=True)
                else:
                    continue

            # Remove the values that have not been seen during fitting or had too few
            # occurrences (Set to NaN)
            df.loc[~df[col_name].isin(values), col_name] = np.nan
            # Perform one-hot-encoding
            for val in values:
                new_col_name = self._gen_col_name(col_name, [val])[0]
                rows_with_val = np.where(df[col_name] == val)[0]
                df[new_col_name] = 0
                df.loc[rows_with_val, new_col_name] = 1

            # Remove original column
            df.drop(col_name, axis=1, inplace=True)
        return df

    def _gen_col_name(self, col_name: str, values: List[str]) -> List[str]:
        """Generate the column names for a column and a list of column values.

        :param col_name: column name
        :param values: column values
        :return: List with the generated column names
        """
        new_col_names = []
        for val in values:
            new_col_names.append(col_name + "=" + val)

        return new_col_names


def compute_temporal_encoding(df, time_col: str, feature, mode):
    """Computes a temporal encoding

    Example: mode="months": the average value of the feature column is computed for
    each month, which is then the encoding for each month.
    Encodings should only be used if the complete spectrum is covered in the training
    set. E.g. when mode = 'months', the training data should contain all months that
    will also be during inference / in the test data set.

    :param df: Dataframe
    :param feature: The feature that is used for temporal encoding
    :param mode: mode="seasons": 4 different encodings for Winter, Spring, Summer and
    Autumn. Each season has 3 months, starting with winter.
    mode="months": 12 different encodings, one for each month
    mode="hours": 24 different encodings, one for each hour of the day.
    mode="weekdays": 7 different encodings, one for each day of the week.
    :return: list with the encoding
    """

    if mode == "seasons":
        winter = [1, 2, 3]
        spring = [4, 5, 6]
        summer = [7, 8, 9]
        fall = [10, 11, 12]

        mean_winter = df[pd.to_datetime(df[time_col]).dt.month.isin(winter)][
            feature
        ].mean()
        mean_spring = df[pd.to_datetime(df[time_col]).dt.month.isin(spring)][
            feature
        ].mean()
        mean_summer = df[pd.to_datetime(df[time_col]).dt.month.isin(summer)][
            feature
        ].mean()
        mean_fall = df[pd.to_datetime(df[time_col]).dt.month.isin(fall)][feature].mean()

        return [mean_winter, mean_spring, mean_summer, mean_fall]

    elif mode == "months":
        means = []
        for i in range(0, 12):
            means.append(
                df[pd.to_datetime(df[time_col]).dt.month == i + 1][feature].mean()
            )
        return means
    elif mode == "hours":
        means = []
        for i in range(0, 24):
            means.append(df[pd.to_datetime(df[time_col]).dt.hour == i][feature].mean())
        return means
    elif mode == "weekdays":

        means = []
        for i in range(0, 7):
            means.append(
                df[pd.to_datetime(df[time_col]).dt.weekday == i][feature].mean()
            )
        return means

    else:
        raise ValueError("mode argument is wrong.")


def add_temporal_encoding(
    df,
    time_col: str,
    train_size=None,
    feature=None,
    mode="seasons",
    encoding=None,
):
    """Adds a column with a temporal encoding.

    Example: mode="months": With the training data defined by train_size, the average
    value of the feature column is computed for each month, which is then the
    encoding  for each month. Then a column to the DataFrame df is added that holds
    the encodings. The column name will be <"temporal_encoding_" + mode>.

    :param df: Dataframe
    :param train_size: percent of data points used to compute the encoding
    :param feature: The feature that is used for temporal encoding
    :param mode: mode="seasons": 4 different encoding for Winter, Spring, Summer and
    Autumn. Each season has 3 months, starting with winter.
    mode="months": 12 different encodings, one for each month
    mode="hours": 24 different encodings, one for each hour of the day.
    mode="weekdays": 7 different encodings, one for each day of the week.
    mode="holidays": encoding 1 if a day is a holiday and 0 if not.
    :param encoding: if an encoding is given, this one is used instead of computing
    it. The
    encoding is a list of the means for the seasons or months.
    :return: Tuple: Dataframe with new column for temporal encoding, array with encoding
    """

    df = df.copy(deep=True)
    if mode == "seasons":
        winter = [1, 2, 3]
        spring = [4, 5, 6]
        summer = [7, 8, 9]
        fall = [10, 11, 12]
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(
                df[:train_len], time_col, feature, mode="seasons"
            )

        def season_mapper(row):
            if pd.to_datetime(row[time_col]).month in winter:
                return means[0]
            elif pd.to_datetime(row[time_col]).month in spring:
                return means[1]
            elif pd.to_datetime(row[time_col]).month in summer:
                return means[2]
            elif pd.to_datetime(row[time_col]).month in fall:
                return means[3]

        df["temporal_encoding_seasons"] = df.apply(
            lambda row: season_mapper(row), axis=1
        )
        return df, means

    elif mode == "months":
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(
                df[:train_len], time_col, feature, mode="months"
            )

        def months_mapper(row):
            return means[row[time_col].month - 1]

        df["temporal_encoding_months"] = df.apply(
            lambda row: months_mapper(row), axis=1
        )
        return df, means

    elif mode == "weekdays":
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(
                df[:train_len], time_col, feature, mode="weekdays"
            )

        def weekdays_wrapper(row):
            return means[row[time_col].weekday()]

        df["temporal_encoding_weekdays"] = df.apply(
            lambda row: weekdays_wrapper(row), axis=1
        )
        return df, means

    elif mode == "hours":
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(
                df[:train_len], time_col, feature, mode="hours"
            )

        def hours_wrapper(row):
            return means[row[time_col].hour]

        df["temporal_encoding_hours"] = df.apply(lambda row: hours_wrapper(row), axis=1)
        return df, means
    else:
        raise ValueError("mode argument is wrong.")


def fill_gaps(df, method="ffill"):
    """Fill gaps in the DataFrame.

    :param df: DataFrame
    :param method: method with which to fill the gaps. Default: "ffill" fills the
    gaps by copying the last observed value.
    :return: DataFrame with the filled gaps
    """
    return df.fillna(method=method)


def smooth_rolling_window(df, features, window_size=5, method="mean", delete_nans=True):
    """filters the features in the DataFrame using a rolling window approach. For
    each datapoint, the n=window_size last datapoints are used to calculate the new
    value for the data point The first and last values of a smoothed feature are set
    to NaN and can those rows be deleted if delete_nans=True

    :param df: the input DataFrame
    :param features: The features that shall be smoothed
    :param window_size: the window size
    :param method: the method that is used. Available: 'median' and 'mean'
    :param delete_nans: if True, the rows where the smoothed features have NaN
    values after smoothing are deleted
    :return: The DataFrame with the smoothed features
    """
    df_copy = df.copy(deep=True)
    for feature in features:
        if method == "mean":
            mean_values = df_copy[feature].rolling(window=window_size).mean().values
            df_copy[feature] = mean_values
        elif method == "median":
            median_values = df_copy[feature].rolling(window=window_size).median().values
            df_copy[feature] = median_values
        else:
            raise ValueError("method must be either 'mean' or 'median'")

    if delete_nans:
        df_copy = df_copy[window_size - 1 :]
    return df_copy
