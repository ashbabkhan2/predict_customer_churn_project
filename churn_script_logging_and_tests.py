"""
In this module we test the function written in churn_library module

Author: Ashbab khan
Date: 12-Jan-2023
"""

import logging
import os
import pytest
import churn_library

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
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # checking the rows and column of our dataframe
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df = data_frame


def test_eda(perform_eda):
    '''
    test eda : this function is testing the perform_eda function
    '''
    # saving variable df back to conftest.py variable using pytest
    data_frame = pytest.df

    try:
        test_eda_df = perform_eda(data_frame)
        logging.info("Testing test_eda : Successful")
    except AttributeError as err:
        logging.error("Testing test_eda : Please enter a dataframe object")
        raise err

    try:
        assert os.path.exists("./images/eda/churn_distribution.png")
        assert os.path.exists("./images/eda/Customer_age_distribution.png")
        assert os.path.exists("./images/eda/heatmap.png")
        assert os.path.exists("./images/eda/marital_status_distribution.png")
        assert os.path.exists(
            "./images/eda/total_transaction_distribution.png")
        logging.info("Testing test_eda : your file saved successfully")
    except AssertionError as err:
        logging.error("Testing test_eda : One of your images is not saved")
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
    data_frame = pytest.df

    try:
        encoder_df = encoder_help(data_frame, cat_columns)
        logging.info("Testing: test_encoder_helper:success")
    except AttributeError as err:
        logging.error("Error in test_encoder_help: please enter a dataframe")
        raise err

    try:
        assert "Gender_Churn" in encoder_df.columns
        assert "Education_Level_Churn" in encoder_df.columns
        assert "Marital_Status_Churn" in encoder_df.columns
        assert "Income_Category_Churn" in encoder_df.columns
        assert "Card_Category_Churn" in encoder_df.columns
        logging.info(
            "test_encoder_helper : all the 5 column successfully created")
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper : error in creating the churn column")
        raise err

    pytest.df = encoder_df


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering : this will test perform_feature_engineering
    '''
    data_frame = pytest.df

    # getting back our splitted value and then checking
    # so that it doesn't contain empty value
    try:
        (input_features_train, input_features_test,target_train,
         target_test) = perform_feature_engineering(data_frame)
        assert input_features_train.shape[0] > 0 and input_features_train.shape[1] > 0
        assert input_features_test.shape[0] > 0 and input_features_test.shape[1] > 0
        assert target_train.shape[0] > 0
        assert target_test.shape[0] > 0
        logging.info("Testing feature Engineering: Success")
    except AssertionError as err:
        logging.error("Error: in feature engineering")
        raise err

    # saving back to our conftest.py file using pytest
    pytest.X_train = input_features_train
    pytest.X_test = input_features_test
    pytest.y_train = target_train
    pytest.y_test = target_test


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

        assert os.path.exists("./images/results/random_forest_results.png")
        assert os.path.exists("./images/results/logistic_regression.png")
        logging.info(
            """Testing test_classification_report :
               random_forest_results and logistic_regression image successfully saved""")
    except AssertionError as err:
        logging.error(
            "Testing test_classification_report : problem in saving images")
        raise err


def test_feature_importance_plot(feature_importance):
    """
    test_feature_importance_plot : testing feature_importance function

    """
    path = "./images/results/feature_importance.png"
    try:
        feature_importance(pytest.cv_rfc, pytest.X_train, path)
        assert os.path.exists(path)
        logging.info(
            "Testing test_feature_importance_plot: feature_importance image successfully saved")
    except AssertionError as err:
        logging.error(
            "Testing test_feature_importance_plot : Image is not saved")
        raise err


def test_train_model(train_model):
    """
    test_train_model : testing the train_model function
    """

    try:
        train_model(
            pytest.X_test,
            pytest.y_test)
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./images/results/logistic_roc_plot.png")
        assert os.path.exists("./images/results/rfc_roc_curve.png")
        logging.info(
            "Testing train model success : Images and model successfully saved")
    except AssertionError as err:
        logging.error(
            "Testing test_train_model : Problem in saving images and models")
        raise err


if __name__ == "__main__":
    test_import(churn_library.import_data)
    test_eda(churn_library.perform_eda)
    test_encoder_helper(churn_library.encoder_helper)
    test_perform_feature_engineering(churn_library.perform_feature_engineering)
    test_classification_report_image(churn_library.classification_report_image)
    test_feature_importance_plot(churn_library.feature_importance_plot)
    test_train_model(churn_library.train_models)
