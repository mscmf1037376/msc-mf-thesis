from typing import Tuple

import pandas as pd


def slice_dataset(data: pd.DataFrame, num_test_samples: int, num_validation_samples: int,
                  overlap_validation_training: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice a given time series dataset into a training, validation and test sets
    The training and validation data are available at the current time t. Test data is meant to be "future" unseen data

    <---------- entire time series length ------------>
    |==========================|==============|=======|
             Training            Validation   |   Test
                                              |
                            "past"   <--  "present"   -->  "future"

    The training and validation data are observable - this is all the data available at the "present" time.
    We can train and validate a model on the training and validation data, then test it on unseen data in the test set
    Since the model is trained and validated with data that is available up to the present time, this type of slicing
    avoids look-ahead bias. Also, the model is validated on the most recent data, which should improve performance on
    the test set as we are training the model to learn the current time series dynamics as closely as possible.

    :param data: returns and features dataframe
    :param num_test_samples: number of samples to leave in the test set
    :param num_validation_samples: number of samples to construct the validation set
    :param overlap_validation_training:
    :return: three dataframes - training, validation and test data
    """

    if num_test_samples < 1 or num_validation_samples < 1:
        raise ValueError('test and validation sets must contain at least one sample')

    if num_test_samples >= len(data):
        raise ValueError('test set cannot be larger than the data')

    if num_validation_samples >= len(data):
        raise ValueError('validation set cannot be larger than the data')

    if num_validation_samples + num_test_samples >= len(data):
        raise ValueError('there is not enough data for the training set')

    test_set = data.tail(num_test_samples)
    validation_set = data.tail(num_test_samples + num_validation_samples).head(num_validation_samples)
    if overlap_validation_training:
        training_set = data.head(len(data) - num_test_samples)
    else:
        training_set = data.head(len(data) - num_validation_samples - num_test_samples)

    return training_set, validation_set, test_set


def rolling_slicer(dataset: pd.DataFrame, time_frame: int, step: int, num_test_samples, num_validation_samples):
    start = 0
    end = start + time_frame

    while end <= len(dataset):
        current_slice = dataset.iloc[start:end]
        yield slice_dataset(current_slice, num_test_samples=num_test_samples,
                            num_validation_samples=num_validation_samples)
        start += step
        end += step
