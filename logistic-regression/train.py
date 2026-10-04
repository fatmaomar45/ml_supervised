import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)

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
# Build a dense feature grid once instead of creating intermediate frames and repeated copies.
grid = np.array(list(itertools.product(*values)), dtype=object)
test_df = pd.DataFrame(grid, columns=keys)

# Keep the original feature order expected by the model.
test_df = test_df[features]

log_preds = log_model.predict(test_df)
test_df['Logistic_Prediction'] = np.where(log_preds == 1, 'Survived', 'Died')

test_df['Pclass'] = test_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})
test_df['Sex'] = test_df['Sex'].map({0: 'Female', 1: 'Male'})
test_df['Embarked'] = test_df['Embarked'].map({0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'})

test_df.to_csv('titanic_logistic_predictions.csv', index=False)
print('Saved to titanic_logistic_predictions.csv')
