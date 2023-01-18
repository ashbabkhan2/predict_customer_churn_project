"""
In this module we test the function written in churn_library module

Author: Ashbab khan
Date: 12-Jan-2023
"""

import churn_library
import pytest
import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# configuring our logging model
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - Used to test import_data function
    Input : import_data function

    Return :
            Nothing

    '''
    # importing data and saving our log message successfull or fail
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # checking the rows and column of our dataframe
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df = df


def test_eda(perform_eda):
    '''
    test eda : this function is testing the perform_eda function
    '''
    # saving variable df back to conftest.py variable using pytest
    df = pytest.df

    try:
        test_eda_df = perform_eda(df)
        logging.info("Testing test_eda : Successful")
    except AttributeError as err:
        logging.error("Testing test_eda : Please enter a dataframe object")
        raise err

    pytest.df = test_eda_df


def test_encoder_helper(encoder_help):
    '''
    test encoder helper : this will test the encoder_help function
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = pytest.df

    try:
        encoder_df = encoder_help(df, cat_columns)
        logging.info("Testing: test_encoder_helper:success")
    except AttributeError as err:
        logging.error("Error in test_encoder_help: please enter a dataframe")
        raise err

    pytest.df = encoder_df

    # checking our cat_column so that it contain 5 values
    try:
        assert len(cat_columns) == 5
        logging.info("Testing test_encoder_helper :success")
    except AssertionError as err:
        logging.error(
            "test_encoder_helper : please include all 5 values in list")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering : this will test perform_feature_engineering
    '''
    df = pytest.df

    # getting back our splitted value and then checking
    # so that it doesn't contain empty value
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert X_train.shape[0] > 0 and X_train.shape[1] > 0
        assert X_test.shape[0] > 0 and X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing feature Engineering: Success")
    except AssertionError as err:
        logging.error("Error: in feature engineering")
        raise err

    # saving back to our conftest.py file using pytest
    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test


def test_classification_report_image(classification_report_image):
    """
    test_classification_report_image : testing classification_report_image

    """

    # passing the required values from the variable store in conftest.py
    try:
        classification_report_image(
            pytest.y_train,
            pytest.y_test,
            pytest.y_train_preds_lr,
            pytest.y_train_preds_rf,
            pytest.y_test_preds_lr,
            pytest.y_test_preds_rf)
        logging.info("test_classification_report:Success")
    except BaseException:
        logging.error("There is some issues in the test_classification_report")


def test_feature_importance_plot(feature_importance):
    """
    test_feature_importance_plot : testing feature_importance function

    """
    path = "./images/results/feature_importance.png"
    try:
        feature_importance(pytest.cv_rfc, pytest.X_train, path)
        logging.info("test_feature_importance_plot: Success")
    except BaseException:
        logging.error("Problem in the test_importance function")


def test_train_model(train_model):
    """
    test_train_model : testing the train_model function
    """

    try:
        train_model(
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test)
        logging.info("testing train model success")
    except BaseException:
        logging.error("Problem in test train model")


if __name__ == "__main__":
    test_import(churn_library.import_data)
    test_eda(churn_library.perform_eda)
    test_encoder_helper(churn_library.encoder_helper)
    test_perform_feature_engineering(churn_library.perform_feature_engineering)
    test_classification_report_image(churn_library.classification_report_image)
    test_feature_importance_plot(churn_library.feature_importance_plot)

    test_train_model(churn_library.train_models)
