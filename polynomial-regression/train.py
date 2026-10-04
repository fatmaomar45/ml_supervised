import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression().fit(X_train_poly, y_train)

scenarios = {
    'Pclass': [1, 2, 3],
    'Sex': [0, 1],
    'Age': list(range(20, 101)),
    'SibSp': list(range(0, 9)),
    'Parch': list(range(0, 7)),
    'Embarked': [0, 1, 2],
    'Fare': [80.0, 20.0, 13.0],
}

keys = list(scenarios.keys())
values = list(scenarios.values())
# Create the scenario grid once; avoid building a large inner list and then reindexing.
grid = np.array(list(itertools.product(*values)), dtype=object)
test_df = pd.DataFrame(grid, columns=keys)

# Preserve the expected feature ordering before applying polynomial features.
test_df = test_df[features]

poly_preds = poly_model.predict(poly.transform(test_df))
test_df['Poly_Prediction'] = np.where(poly_preds >= 0.5, 'Survived', 'Died')

test_df['Pclass'] = test_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})
test_df['Sex'] = test_df['Sex'].map({0: 'Female', 1: 'Male'})
test_df['Embarked'] = test_df['Embarked'].map({0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'})

test_df.to_csv('titanic_polynomial_predictions.csv', index=False)
