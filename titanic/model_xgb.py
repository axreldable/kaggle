import xgboost as xgb
from sklearn.metrics import accuracy_score

from titanic.data import get_titanic_data, SEED
from titanic.features import prepare_features_xgb

train_df, test_df = get_titanic_data()
X_train, X_test, y_train, y_test = prepare_features_xgb(train_df, 'Survived')

model = xgb.XGBClassifier(random_state=SEED)
model.fit(X_train, y_train)
print(model)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print('accuracy_score: ', acc)
# 0.8071748878923767
