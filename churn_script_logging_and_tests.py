"""
In this module we test the function written in churn_library module

Author: Ashbab khan
Date: 12-Jan-2023
"""

# import os
import churn_library
import pytest
import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# import pandas as pd
# import churn_library_solution as cls
# global_df = churn_library.import_data("./data/bank_data.csv")

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# global_df = churn_library.import_data("")

@pytest.fixture(scope="module")
def path():
	data_path = "./data/bank_data.csv"
	yield data_path

@pytest.fixture(scope="module")
def test_import(path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = churn_library.import_data(path)
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
		logging.error(
		    "Testing import_data: The file doesn't appear to have rows and columns")
# 		raise err

	yield df

# @pytest.fixture(scope="module")
def test_eda(test_import):
	'''
	test perform eda function
	'''
	# df = test_import(churn_library.import_data)
	try:
		df_a = churn_library.perform_eda(test_import)
		print(df_a.columns)
		logging.info("Testing test_eda : Successful")
	except AttributeError:
		logging.error("Testing test_eda : Please enter a dataframe object")

	return df_a	
# 		raise err



# def test_encoder_helper(df):
# 	'''
# 	test encoder helper
# 	'''
# 	cat_columns = [
#     'Gender',
#     'Education_Level',
#     'Marital_Status',
#     'Income_Category',
#     'Card_Category'
#     ]

# 	# df = test_import(churn_library.import_data)
# 	try:
# 		# encoder_helper(global_df,cat_columns)
# 		encoder_df = churn_library.encoder_helper(df,cat_columns)
# 		logging.info("Testing: test_encoder_helper:success")
# 	except AttributeError:
# 		logging.error("Error in test_encoder_help: please enter a dataframe")

# 	return encoder_df	


	# try:
	# 	assert len(cat_columns) == 5
	# 	logging.info("test_encoder_helper :success")
	# except AssertionError:
	# 	logging.error("test_encoder_helper : please include all 5 values in list")



    # 	encoder_helper(df,cat_columns)
	# 	logging.info("Testing: test_encoder_helper:success")
	# except AssertionError:
	#     logging.error("Error: ")

# def test_perform_feature_engineering(test_encoder_df):
# 	'''
# 	test perform_feature_engineering
# 	'''
# 	try:
# 		data = churn_library.perform_feature_engineering(test_encoder_df)
# 		logging.info("Testing feature Engineering: Success")
# 	except:
# 		logging.error("Error: in feature engineering")







# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


# if __name__ == "__main__":
# 	test_eda()

	# df = test_import(churn_library.import_data)
	# test_eda_df = test_eda(df)
	# test_encoder_df = test_encoder_helper(test_eda_df)
	# test_perform_feature_engineering(test_encoder_df)





    # global_df = churn_library.import_data("./data/bank_data.csv")
    # test_import(import_data)
    # test_eda()
    # test_encoder_helper(churn_library.encoder_helper)
    # pass








