from sklearn.ensemble import RandomForestClassifier
import pickle

with open("/content/credit.pkl", "rb") as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)

random_forest = RandomForestClassifier(
    n_estimators=45, criterion="entropy", random_state=0
)
random_forest.fit(x_credit_training, y_credit_training)

forecast = random_forest.predict(x_credit_test)
print(forecast)
