from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

base_census = pd.read_csv("../census.csv")

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

lable_encode_workclass = LabelEncoder()
lable_encode_education = LabelEncoder()
lable_encode_marital_status = LabelEncoder()
lable_encode_occupation = LabelEncoder()
label_encode_relationship = LabelEncoder()
label_encode_race = LabelEncoder()
label_encode_sex = LabelEncoder()
label_encode_native_country = LabelEncoder()

x_census[:, 1] = lable_encode_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = lable_encode_education.fit_transform(x_census[:, 3])
x_census[:, 5] = lable_encode_marital_status.fit_transform(x_census[:, 5])
x_census[:, 6] = lable_encode_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encode_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encode_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encode_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encode_native_country.fit_transform(x_census[:, 13])

onehotencoder_census = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder="passthrough",
)
x_census = onehotencoder_census.fit_transform(x_census).toarray()
scale_census = StandardScaler()
x_census = scale_census.fit_transform(x_census)

from sklearn.model_selection import train_test_split

(
    x_census_treinamento,
    x_census_test,
    y_census_treinamento,
    y_census_test,
) = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

# with open("../census.pkl", mode="wb") as f:
#     pickle.dump(
#         [x_census_treinamento, y_census_treinamento, x_census_test, y_census_test], f
#     )

with open("../census.pkl", "rb") as f:
    (
        x_census_treinamento,
        y_census_treinamento,
        x_census_test,
        y_census_test,
    ) = pickle.load(f)

arvore_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_census.fit(x_census_treinamento, y_census_treinamento)

previsao = arvore_census.predict(x_census_test)
previsao
accuracy_score(y_census_test, previsao)
