import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
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


def apply_transformation(dataframe, transformation):
    """Applies transformation rules to the dataframe"""
    for column, rules in transformation.items():
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(rules)
    return dataframe


def get_transform_dict():
    """Returns a dictionary of transformation rules"""

    trf_dict = {
        "purpose": {
            0: "others",
            1: "car",
            2: "car",
            3: "domestic appliances",
            4: "domestic appliances",
            5: "domestic appliances",
            6: "repairs",
            7: "education",
            8: "vacation",
            9: "retraining",
            10: "business",
        },
        "personal_status_sex": {1: "male", 2: "female", 3: "male", 4: "female"},
        "property": {
            1: "unknown / no property",
            2: "car or other",
            3: "building soc. savings agr./life insurance",
            4: "real estate",
        },
        "foreign_worker": {1: 1, 2: 0},
    }
    return trf_dict


def train_test_xgb(df, target):
    """Trains and Tests the XGBoost model"""
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=0.3, random_state=42
    )

    # Define the categorical columns
    categorical_cols = ["purpose", "personal_status_sex", "property"]

    # Define the preprocessing steps for categorical columns
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combine the preprocessing steps with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_cols)]
    )

    # Instantiate the XGBoost classifier
    classifier = XGBClassifier()

    # Create the pipeline by combining preprocessor and classifier
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Perform cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=kfold, scoring="roc_auc")

    # Train the classifier on the full training set
    pipeline.fit(x_train, y_train)

    # Calculate the test score
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_score = roc_auc_score(y_test, test_proba)

    print("Test set ROC-AUC:", test_score)

    # Combine the y_test and test_proba into a dataframe
    results = x_test.copy()
    results["y_true"] = y_test
    results["y_pred"] = test_proba

    # Apply a threshold to the probability to predict the class labels
    results["y_pred_class"] = results["y_pred"].apply(lambda x: 1 if x > 0.5 else 0)

    return pipeline, results
