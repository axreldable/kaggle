from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from titanic.data import SEED, get_titanic_data
from titanic.features import get_feature_transformer, prepare_features_2
from titanic.submission import generate_and_save_submission

train_df, test_df = get_titanic_data()

# X_train, X_test, y_train, y_test = prepare_features_1(train_df)
feature_transformer = get_feature_transformer(train_df)
X_train, X_test, y_train, y_test = prepare_features_2(train_df, feature_transformer)

model = RandomForestClassifier(criterion='gini',
                               n_estimators=1750,
                               max_depth=7,
                               min_samples_split=6,
                               min_samples_leaf=6,
                               max_features='auto',
                               oob_score=True,
                               random_state=SEED,
                               n_jobs=-1,
                               verbose=1)
model.fit(X_train, y_train)
print(model)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print('accuracy_score: ', acc)

X_rez = feature_transformer.transform(test_df)
rez_preds = model.predict(X_rez)

generate_and_save_submission(test_df, rez_preds, 'submissions_2.csv')
