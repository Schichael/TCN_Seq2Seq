import pandas as pd

# import holidays


def compute_temporal_encoding(df, feature, mode):
    """Computes a temporal encoding

    Example: mode="months": the average value of the feature column is computed for
    each month, which is then the encoding for each month.

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

        mean_winter = df[pd.to_datetime(df["date / time"]).dt.month.isin(winter)][
            feature
        ].mean()
        mean_spring = df[pd.to_datetime(df["date / time"]).dt.month.isin(spring)][
            feature
        ].mean()
        mean_summer = df[pd.to_datetime(df["date / time"]).dt.month.isin(summer)][
            feature
        ].mean()
        mean_fall = df[pd.to_datetime(df["date / time"]).dt.month.isin(fall)][
            feature
        ].mean()

        return [mean_winter, mean_spring, mean_summer, mean_fall]

    elif mode == "months":
        means = []
        for i in range(0, 12):
            means.append(
                df[pd.to_datetime(df["date / time"]).dt.month == i + 1][feature].mean()
            )
        return means
    elif mode == "hours":
        means = []
        for i in range(0, 24):
            means.append(
                df[pd.to_datetime(df["date / time"]).dt.hour == i][feature].mean()
            )
        return means
    elif mode == "weekdays":

        means = []
        for i in range(0, 7):
            means.append(
                df[pd.to_datetime(df["date / time"]).dt.weekday == i][feature].mean()
            )
        return means

    else:
        raise ValueError("mode argument is wrong.")


def add_temporal_encoding(
    df,
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
            means = compute_temporal_encoding(df[:train_len], feature, mode="seasons")

        def season_mapper(row):
            if pd.to_datetime(row["date / time"]).month in winter:
                return means[0]
            elif pd.to_datetime(row["date / time"]).month in spring:
                return means[1]
            elif pd.to_datetime(row["date / time"]).month in summer:
                return means[2]
            elif pd.to_datetime(row["date / time"]).month in fall:
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
            means = compute_temporal_encoding(df[:train_len], feature, mode="months")

        def months_mapper(row):
            return means[row["date / time"].month - 1]

        df["temporal_encoding_months"] = df.apply(
            lambda row: months_mapper(row), axis=1
        )
        return df, means

    elif mode == "weekdays":
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(df[:train_len], feature, mode="weekdays")

        def weekdays_wrapper(row):
            return means[row["date / time"].weekday()]

        df["temporal_encoding_weekdays"] = df.apply(
            lambda row: weekdays_wrapper(row), axis=1
        )
        return df, means

    elif mode == "hours":
        if encoding:
            means = encoding
        else:
            train_len = int(df.shape[0] * train_size)
            means = compute_temporal_encoding(df[:train_len], feature, mode="hours")

        def hours_wrapper(row):
            return means[row["date / time"].hour]

        df["temporal_encoding_hours"] = df.apply(lambda row: hours_wrapper(row), axis=1)
        return df, means

    # elif mode == "holidays":
    #    german_holidays = holidays.DE()
    #
    #        def holidays_wrapper(row):
    #            if row["date / time"] in german_holidays:
    #                return 1
    #            else:
    #                return 0
    #
    #        df["temporal_encoding_holidays"] = df.apply(
    #            lambda row: holidays_wrapper(row), axis=1
    #        )
    #        return df, [0, 1]

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


def remove_inactive_days(df):
    """Removes the days where "MeteoViva_Switch - MeteoViva inactive (Logging)" == 1
    or MeteoViva_Schalter - MeteoViva_aktiv (Logging) == 0

    :param df: input DataFrame
    :return: Pandas DataFrame with removed days
    """
    df_copy = df.copy(deep=True)
    datetime_inactive = df_copy[
        (df_copy["MeteoViva_Switch - MeteoViva inactive (Logging)"] == 1)
        | (df_copy["MeteoViva_Schalter - MeteoViva_aktiv (Logging)"] == 0)
    ]["date / time"]
    days_inactive = datetime_inactive.dt.date.unique()
    df_inactive_removed = df_copy[~df_copy["date / time"].dt.date.isin(days_inactive)]

    return df_inactive_removed


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
