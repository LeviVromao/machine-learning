from sklearn.ensemble import RandomForestClassifier

with open("/content/census.pkl", "rb") as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)

random_forest_census = RandomForestClassifier(
    n_estimators=40, criterion="entropy", random_state=0
)
random_forest.fit(x_census_training, y_census_training)

forecast = random_forest.predict(x_census_test)
forecast
