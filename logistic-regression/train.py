import itertools
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
    'Embarked': [0, 1, 2]
}
keys, values = zip(*scenarios.items())
grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
test_df = pd.DataFrame(grid)
test_df['Fare'] = test_df['Pclass'].map({1: 80.0, 2: 20.0, 3: 13.0})
test_df = test_df[features]


chunk_size = 50000
log_preds = []
for i in range(0, len(test_df), chunk_size):
    chunk = test_df.iloc[i:i+chunk_size]
    log_preds.extend(log_model.predict(chunk))

test_df['Logistic_Prediction'] = ["Survived" if p == 1 else "Died" for p in log_preds]


test_df['Pclass'] = test_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})
test_df['Sex'] = test_df['Sex'].map({0: 'Female', 1: 'Male'})
test_df['Embarked'] = test_df['Embarked'].map({0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'})


test_df.to_csv('titanic_logistic_predictions.csv', index=False)
print("Saved to titanic_logistic_predictions.csv")