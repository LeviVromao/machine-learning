import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# from yellowbrick.classifier import ConfussionMatrix
import pickle

with open("../credit.pkl", mode="rb") as f:
    (
        x_credit_treinamento,
        y_credit_treinamento,
        x_credit_test,
        y_credit_test,
    ) = pickle.load(f)

arvore_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = arvore_credit.predict(x_credit_test)
print(previsoes)
# print(y_credit_test)

# cm = ConfussionMatrix(arvore_credit)
# cm.fit(x_credit_treinamento, y_credit_treinamento)
# cm.score(x_credit_test, y_credit_test)
# print(accuracy_score(y_credit_test, previsoes))
