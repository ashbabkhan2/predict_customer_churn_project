"""
Author : ashbab
Date : 18 Jan 2023

this file will store variables which will be used
all of the files in this directory the main use of this
file is that it is accessible to all the files of this directory
"""

import pytest


def df_plugin():
    """
    Input : None

    Return : None
    """
    return None


def pytest_configure():
    """
    this function is storing variables so that it is
    used throughout the directory.

    Input : None

    Return : None

    """
    pytest.df = df_plugin()
    pytest.y_train = df_plugin()
    pytest.y_test = df_plugin()
    pytest.X_train = df_plugin()
    pytest.X_test = df_plugin()
    pytest.y_train_preds_lr = df_plugin()
    pytest.y_train_preds_rf = df_plugin()
    pytest.y_test_preds_lr = df_plugin()
    pytest.y_test_preds_rf = df_plugin()
    pytest.cv_rfc = df_plugin()
    pytest.lrc = df_plugin()

