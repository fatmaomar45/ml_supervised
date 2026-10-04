import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

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
# Build the full scenario grid once to avoid repeated DataFrame transformations.
grid = np.array(list(itertools.product(*values)), dtype=object)
test_df = pd.DataFrame(grid, columns=keys)

test_df = test_df[features]

rf_preds = rf_model.predict(test_df)
test_df['RF_Prediction'] = np.where(rf_preds == 1, 'Survived', 'Died')

test_df['Pclass'] = test_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})
test_df['Sex'] = test_df['Sex'].map({0: 'Female', 1: 'Male'})
test_df['Embarked'] = test_df['Embarked'].map({0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'})

test_df.to_csv('titanic_randomforest_predictions.csv', index=False)
print('Saved to titanic_rf_predictions.csv')
