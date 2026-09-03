# Machine Learning - Titanic Survival Prediction

Three models trained on the Titanic dataset to predict passenger survival:

- **Logistic Regression** - linear baseline model
- **Random Forest** - ensemble tree-based model  
- **Polynomial Regression** - linear regression with polynomial features

## Structure

```
machine-learning/
├── logistic-regression/
│   ├── train.py
│   └── titanic_logistic_predictions.csv
├── random-forest/
│   ├── train.py
│   └── titanic_randomforest_predictions.csv
└── polynomial-regression/
    ├── train.py
    └── titanic_polynomial_predictions.csv
```

## Running

```bash
cd logistic-regression && python train.py
cd ../random-forest && python train.py
cd ../polynomial-regression && python train.py
```
