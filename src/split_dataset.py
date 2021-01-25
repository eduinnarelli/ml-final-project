'''
Auxiliar function for splitting dataset into train, validation and test, used
at 'scripts/prepare_data.py'.
'''
from typing import TypeVar

from sklearn.model_selection import train_test_split

DataFrame = TypeVar('pandas.core.frame.DataFrame')


def split_dataset(
    df: DataFrame,
    test_size: float,
    valid_size: float,
    target_col_name: str,
    random_state: int
) -> DataFrame:
    '''
    Split dataset into train, validation and test.

    Args:
        df: dataset stored as a pandas dataframe.
        test_size: fraction of the dataset that will be used for testing.
        valid_size: fraction of the dataset that will be used for validation.
        target_col_name: name of the target column in the dataframe.
        random_state: controls the shuffling applied to the data before
            applying the split. Pass an int for reproducible output across
            multiple function calls.

    Returns:
        Tuple with the splitted dataset (train, test, valid).
    '''

    # Check if test/valid sizes are in the range [0, 1]
    assert 0.0 < test_size < 1.0, 'test_size must be in the range (0, 1)'
    assert 0.0 <= valid_size < 1.0, 'valid_size must be in the range [0, 1]'

    # Make sure the dataframe provided has a `target_col_name` column
    assert target_col_name in df.columns, \
        f'The dataframe must have a `{target_col_name}` column'

    # First, split dataset into train and test
    train, test = train_test_split(
        df,
        train_size=(1.0 - test_size),
        random_state=random_state,
        # According to the sklearn docs, if shuffle=False then stratify must
        # be None.
        shuffle=True,
        stratify=df[target_col_name],
    )
    train, test = train.copy(), test.copy()

    # Check if all target classes are represented in both train and test
    # samples
    targets_set = set(df[target_col_name].unique().tolist())
    targets_set_train = set(train[target_col_name].unique().tolist())
    targets_set_test = set(test[target_col_name].unique().tolist())
    assert targets_set == targets_set_train, \
        'Targets in the train sample does not match targets in the full' \
        'dataset'
    assert targets_set == targets_set_test, \
        'Targets in the test sample does not match targets in the full dataset'

    # Check if train and test datasets do not overlap
    assert set(list(train.index)).isdisjoint(set(list(test.index))), \
        'Train and test samples have common entries'

    if valid_size > 0.0:

        # Then, split train dataset into train and validation
        train, valid = train_test_split(train,
                                        train_size=(1.0 - valid_size),
                                        random_state=random_state,
                                        shuffle=True,
                                        stratify=train[target_col_name])
        train, valid = train.copy(), valid.copy()

        # Check if all target classes are represented in both train and valid
        # samples
        targets_set_valid = set(valid[target_col_name].unique().tolist())
        assert targets_set == targets_set_valid, \
            'Targets in the valid sample does not match targets in the full' \
            'dataset'

        # Check if train and valid datasets do not overlap
        assert set(list(train.index)).isdisjoint(set(list(valid.index))), \
            'Train and valid samples have common entries'

    else:
        valid = None

    print('-->  Train size:', len(train))
    print('-->  Valid size:', len(valid) if valid is not None else 0)
    print('-->  Test size:', len(test))

    return train, test, valid
