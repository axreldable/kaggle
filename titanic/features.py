import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from titanic.data import SEED

categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']  # Embarked - nulls (2)
numerical_features = ['Age', 'Fare']  # Embarked - nulls
drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin']
target = 'Survived'


def prepare_features_1(X: pd.DataFrame, cat_cols_list=None, num_cols_list=None):
    X = X.copy()
    y = X[target]
    X.drop(target, axis='columns', inplace=True)

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    # todo: refactor categorical_pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    # todo: refactor numeric_pipeline
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
    )
    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )
    X_processed = full_processor.fit_transform(X)

    # todo: check why we use it here?
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, stratify=y_processed, random_state=SEED
    )

    return X_train, X_test, y_train, y_test


# todo: try this: https://ubc-cs.github.io/cpsc330/lectures/06_column-transformer-text-feats.html
def get_feature_transformer(X: pd.DataFrame):
    X = X.copy()
    X.drop(target, axis='columns', inplace=True)

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ]
    )

    full_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    full_transformer.fit(X)
    return full_transformer


def prepare_features_2(X: pd.DataFrame, feature_transformer: ColumnTransformer, is_debug=True):
    if is_debug:
        print(feature_transformer.get_feature_names_out())

    X_processed = feature_transformer.transform(X)

    y = X[target]
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, stratify=y_processed, random_state=SEED
    )

    return X_train, X_test, y_train, y_test


def age_processing(train_df, test_df):
    data = [train_df, test_df]

    # todo: fill age base on other features (build a model for this?)
    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    print(train_df["Age"].isnull().sum())
    print(test_df["Age"].isnull().sum())


def embarked_processing(train_df):
    common_value = 'S'
    train_df["Embarked"] = train_df["Embarked"].fillna(common_value)


def data_processing(train_df, test_df):
    age_processing(train_df, test_df)
    embarked_processing(train_df)
