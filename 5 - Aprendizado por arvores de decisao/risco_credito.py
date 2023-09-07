from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

with open("../risco_credito.pkl", "rb") as f:
    x_risk_credit, y_risk_credit = pickle.load(f)

arvore_risk_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_risk_credit.fit(x_risk_credit, y_risk_credit)
prediction = arvore_risk_credit.predict([[2, 0, 0, 2], [2, 0, 0, 0]])

print(prediction)
