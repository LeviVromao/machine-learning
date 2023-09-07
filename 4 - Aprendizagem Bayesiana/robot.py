import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv("../credit_data.csv")

base_credit["age"][base_credit["age"] > 0].mean()
base_credit["age"].fillna(base_credit["age"].mean(), inplace=True)

# print(base_credit)

x_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4]
(
    x_credit_treinamento,
    x_credit_test,
    y_credit_treinamento,
    y_credit_test,
) = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

with open("./credit.pkl", mode="wb") as f:
    pickle.dump(
        [x_credit_treinamento, y_credit_treinamento, x_credit_test, y_credit_test], f
    )

naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = naive_credit_data.predict(x_credit_test)
# print(previsoes)

print(classification_report(y_credit_test, previsoes))
