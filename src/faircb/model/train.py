import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def read_csv_data(file_path):
    """Reads the csv data from the file path"""
    data = pd.read_csv(file_path)
    return data


def apply_transformation(dataframe):
    """Applies transformation rules to the dataframe"""
    # Get the transformation rules
    transformation = get_transform_dict()
    for column, rules in transformation.items():
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(rules)
    # Rename the columns
    dataframe = rename_columns(dataframe)

    return dataframe


def rename_columns(dataframe):
    """Renames the columns of the dataframe"""
    # Create a renaming dictionary
    rename_dict = {
        "purpose": "isPersonalLoan",
        "personal_status_sex": "isMale",
        "property": "hasProperty",
        "foreign_worker": "isForeignWorker",
    }
    # Rename the columns
    dataframe.rename(columns=rename_dict, inplace=True)
    return dataframe


def get_transform_dict():
    """Returns a dictionary of transformation rules"""

    trf_dict = {
        "purpose": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 0},
        "personal_status_sex": {1: 1, 2: 0, 3: 1, 4: 0},
        "property": {
            1: 0,
            2: 1,
            3: 1,
            4: 1,
        },
        "foreign_worker": {1: 1, 2: 0},
        "credit_risk": {1: 0, 0: 1},
    }
    return trf_dict


def tune_xgb_classifier_randomized(X, y, param_grid, n_iter=10, cv=5):
    """Tunes the XGBoost classifier using RandomizedSearchCV"""

    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier()

    # Create RandomizedSearchCV object with the specified parameter grid and cross-validation
    random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=n_iter, cv=cv)

    # Fit the random search object to the training data
    random_search.fit(X, y)

    # Print the best parameters and best score found during the random search
    print("Best parameters found: ", random_search.best_params_)
    print("Best score found: ", random_search.best_score_)

    # Return the fitted random search object
    return random_search.best_params_


def tune_xgb_classifier_with_early_stopping(
    X,
    y,
    param_grid,
    n_iter=10,
    cv=5,
    early_stopping_rounds=10,
    test_size=0.2,
    random_state=42,
):
    """Tunes the XGBoost classifier using RandomizedSearchCV with early stopping"""
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier()

    # Create RandomizedSearchCV object with the specified parameter grid and cross-validation
    random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=n_iter, cv=cv)

    # Train the model with early stopping
    random_search.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    # Print the best parameters and best score found during the random search
    print("Best parameters found: ", random_search.best_params_)
    print("Best score found: ", random_search.best_score_)

    # Return the fitted random search object
    return random_search.best_params_


def _select_columns_ignore_missing(df, columns):
    # Filter and select only the existing columns from the DataFrame
    existing_columns = [col for col in columns if col in df.columns]

    # Return a new DataFrame with the selected columns
    return df[existing_columns]


def train_test_xgb(df, target, keep_cols=[], tune=False):
    """Trains and Tests the XGBoost model"""

    # Select only the columns to keep, ignore if not present

    df = _select_columns_ignore_missing(df, keep_cols)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=0.3, random_state=42
    )

    # Assuming you have your features stored in X and your target variable in y

    # Define the parameter grid to search over
    param_grid = {
        "learning_rate": [0.1, 0.01, 0.001],
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
    }

    if tune:
        # Call the function to tune the XGBoost classifier using randomized search
        best_params = tune_xgb_classifier_with_early_stopping(
            x_train, y_train, param_grid, n_iter=20
        )
    else:
        best_params = {
            "subsample": 0.7,
            "n_estimators": 50,
            "max_depth": 2,
            "learning_rate": 0.1,
            "gamma": 0,
            "colsample_bytree": 0.9,
        }

    classifier = XGBClassifier(**best_params)

    # Perform cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        classifier, x_train, y_train, cv=kfold, scoring="roc_auc"
    )

    # Train the classifier on the full training set
    classifier.fit(x_train, y_train)
    calibrated_model = CalibratedClassifierCV(classifier, cv="prefit", method="sigmoid")
    calibrated_model.fit(x_train, y_train)

    # Calculate the test score
    train_proba = calibrated_model.predict_proba(x_train)[:, 1]
    test_proba = calibrated_model.predict_proba(x_test)[:, 1]
    train_score = roc_auc_score(y_train, train_proba)
    test_score = roc_auc_score(y_test, test_proba)

    print("Train set ROC-AUC:", train_score)
    print("CV ROC-AUC:", np.mean(cv_scores))
    print("Test set ROC-AUC:", test_score)

    # Combine the y_test and test_proba into a dataframe
    results = x_test.copy()
    results["label_value"] = y_test.values
    results["y_pred"] = test_proba

    # Apply a threshold to the probability to predict the class labels
    median_proba = results["y_pred"].median()
    results["score"] = results["y_pred"].apply(lambda x: 1 if x > median_proba else 0)

    data_dict = {
        "xtrain": x_train,
        "xtest": x_test,
        "ytrain": y_train,
        "ytest": y_test,
        "results": results,
    }
    return classifier, data_dict


def calibrate_xgboost_probs(Xxgb_model):
    """Calibrates the predicted probabilities of an XGBoost model"""

    # Perform Platt scaling calibration

    calibrated_model.fit(X_train, y_train)

    # Calibrate the predicted probabilities
    calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]

    return calibrated_probs
