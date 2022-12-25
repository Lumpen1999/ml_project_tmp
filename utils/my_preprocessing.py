import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder


def standardization(X_train: pd.DataFrame, X_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    if type(X_train) is pd.core.frame.Series:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    num_cols = [c for c in X_train.columns if  X_train[c].dtype in ['float']]

    if len(num_cols) > 0:
        sc = StandardScaler().fit(X_train[num_cols])
        scaled_train = pd.DataFrame(sc.transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
        scaled_test = pd.DataFrame(sc.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
        X_train.update(scaled_train)
        X_test.update(scaled_test)
        print(f'Standardization for {len(num_cols)} columns')

    return X_train, X_test


def _df_merge_for_encoding(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:

    # df1_col_num = df1.values.shape[1]
    # df1_row_num = df1.values.shape[0]

    # df2_col_num = df2.values.shape[1]
    # df2_row_num = df2.values.shape[0]

    df1_row_num, df1_col_num =  df1.values.shape
    df2_row_num, df2_col_num =  df2.values.shape

    df1 = pd.concat([df1, df2], axis=1)

    new_df_col_num = df1.values.shape[1]
    new_df_row_num = df1.values.shape[0]

    assert(df1_col_num + df2_col_num == new_df_col_num)
    assert(df1_row_num == df2_row_num == new_df_row_num)

    return df1


def one_hot_encoding(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_col_index: list) -> (pd.DataFrame, pd.DataFrame):

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_cols = [c for i, c in enumerate(X_train.columns) if i in cat_col_index]

    if len(cat_cols) > 0:

        X_train_new_cols = encoder.fit_transform(X_train[cat_cols].values)
        X_test_new_cols = encoder.transform(X_test[cat_cols].values)

        new_cols_names = encoder.get_feature_names_out(cat_cols)
        X_train_new_cols = pd.DataFrame(X_train_new_cols, columns=new_cols_names, index=X_train.index)
        X_test_new_cols = pd.DataFrame(X_test_new_cols, columns=new_cols_names, index=X_test.index)

        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)

        X_train = _df_merge_for_encoding(X_train, X_train_new_cols)
        X_test = _df_merge_for_encoding(X_test, X_test_new_cols)

        print(f'One hot encoding for {len(cat_cols)} columns')

    return X_train, X_test


def label_encoding(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_col_index: list) -> (pd.DataFrame, pd.DataFrame):

    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        # encoded_missing_value=np.nan,
    )

    cat_cols = [c for i, c in enumerate(X_train.columns) if i in cat_col_index]

    if len(cat_cols) > 0:

        X_train_new_cols = encoder.fit_transform(X_train[cat_cols].values)
        X_train_new_cols = pd.DataFrame(X_train_new_cols, columns=cat_cols, index=X_train.index) 
        X_train.update(X_train_new_cols)

        X_test_new_cols = encoder.transform(X_test[cat_cols].values)
        X_test_new_cols = pd.DataFrame(X_test_new_cols, columns=cat_cols, index=X_test.index)
        X_test.update(X_test_new_cols)

        print(f'Label encoding for {len(cat_cols)} columns')

    return X_train, X_test


def label_encoding_for_target(y_train: pd.Series, y_test: pd.Series) -> (pd.Series, pd.Series):
    encoder = LabelEncoder(
        # handle_unknown='use_encoded_value',
        # unknown_value=-1,
        # encoded_missing_value=np.nan,
    )
    y_train = pd.Series(encoder.fit_transform(y_train))
    y_test = pd.Series(encoder.transform(y_test))
    return y_train, y_test

