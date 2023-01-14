"""
In this module we test the function written in churn_library module

Author: Ashbab khan 
Date: 12-Jan-2023
"""

# import os
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pandas as pd
import logging
# import churn_library_solution as cls
import churn_library

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# global_df = []
def test_import(import_data):
	'''
	test data import - this example is completed for     you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv");
# 		global_df = df
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError:
		logging.error("Testing import_eda: The file wasn't found")
# 		raise err

# 	return df

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
# 		raise err

	return df
        


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	df = test_import(churn_library.import_data)
	try:
		perform_eda(df);
		logging.info("Testing test_eda : Successful")
	except AttributeError:
		logging.error("Testing test_eda : Please enter a dataframe object")
# 		raise err
        
def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
    test_import(churn_library.import_data)
    test_eda(churn_library.perform_eda)
    pass








