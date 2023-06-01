import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from titanic.data import SEED


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


def age_processing_1(train_df, test_df):
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


def age_processing_2(train_df, test_df):
    df_all = concat_df(train_df, test_df)

    age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

    for pclass in range(1, 4):
        for sex in ['female', 'male']:
            print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
    print('Median age of all passengers: {}'.format(df_all['Age'].median()))

    # Filling the missing values in Age with the medians of Sex and Pclass groups
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    print(train_df["Age"].isnull().sum())
    print(test_df["Age"].isnull().sum())


def embarked_processing(train_df):
    common_value = 'S'
    train_df["Embarked"] = train_df["Embarked"].fillna(common_value)


def data_processing(train_df, test_df):
    age_processing(train_df, test_df)
    embarked_processing(train_df)


categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Ticket', 'Cabin']  # Embarked - nulls (2)
numerical_features = ['Age', 'Fare']  # Embarked - nulls
drop_features = ['PassengerId', 'Name']
target = 'Survived'


def feature_importance(model, feature_names):
    forest_importances = pd.Series(model.feature_importances_, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()


def generate_and_save_submission(test_df: pd.DataFrame, rez_preds, rez_file_name: str):
    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission_df['PassengerId'] = test_df['PassengerId']
    submission_df['Survived'] = rez_preds
    submission_df.to_csv(rez_file_name, header=True, index=False)
    submission_df.head(10)


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
