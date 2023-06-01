import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

target = 'Survived'


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


def generate_features(data_df):
    data_df['family_size'] = data_df['SibSp'] + data_df['Parch']
    data_df['name_length'] = data_df['Name'].apply(len)
    data_df['is_alone'] = 0
    data_df.loc[data_df['family_size'] == 1, 'is_alone'] = 1

    data_df['cabin'] = data_df['Cabin'].str[:1]

    data_df['title'] = 0
    data_df['title'] = data_df.Name.str.extract('([A-Za-z]+)\.')
    data_df['title'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mrs'],
        inplace=True
    )

    data_df.loc[(data_df.Age.isnull()) & (data_df.title == 'Mr'), 'Age'] = data_df.Age[data_df.title == 'Mr'].mean()
    data_df.loc[(data_df.Age.isnull()) & (data_df.title == 'Mrs'), 'Age'] = data_df.Age[data_df.title == 'Mrs'].mean()
    data_df.loc[(data_df.Age.isnull()) & (data_df.title == 'Master'), 'Age'] = data_df.Age[
        data_df.title == 'Master'].mean()
    data_df.loc[(data_df.Age.isnull()) & (data_df.title == 'Miss'), 'Age'] = data_df.Age[data_df.title == 'Miss'].mean()
    data_df.loc[(data_df.Age.isnull()) & (data_df.title == 'Other'), 'Age'] = data_df.Age[
        data_df.title == 'Other'].mean()

    data_df.loc[data_df.Ticket.str.isdigit(), 'ticket_class'] = 1
    data_df.loc[~data_df.Ticket.str.isdigit(), 'ticket_class'] = 0
    data_df['ticket_class'] = data_df['ticket_class'].apply(int)

    data_df = data_df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'])
    return data_df


def get_feature_transformer(X: pd.DataFrame):
    X = X.copy()

    categorical_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'family_size', 'is_alone', 'cabin', 'title',
                            'ticket_class']
    numerical_features = ['Age', 'Fare']

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
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


def generate_and_save_submission(test_df: pd.DataFrame, rez_preds, rez_file_name: str):
    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission_df['PassengerId'] = test_df['PassengerId']
    submission_df['Survived'] = rez_preds
    submission_df.to_csv(rez_file_name, header=True, index=False)
    submission_df.head(10)
