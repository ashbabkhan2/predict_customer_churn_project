# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

### Table of contents
#### [ Project Description ](#project-description)
#### [ Technology used ](#technology-used)
#### [ Packages and library used in this project ](#packages-and-library-used-in-this-project)
#### [ Files and data description ](#files-and-data-description)
#### [ Essential files of this project ](#essential-files-of-this-project)
#### [ Running Files ](#running-files)

## Project Description
In this project, we have to test a churn_notebook.ipynb file we make two major files using the notebook file which are churn_library.py and the second is churn_script_logging_and_tests.py 
churn_library.py file.

### Technology used 

#### ✅ Python 
#### ✅ Machine Learning

### Packages and library used in this project 

##### ✅ scikit-learn 
##### ✅ joblib 
##### ✅ pandas 
##### ✅ numpy 
##### ✅ pytest 
##### ✅ matplotlib 
##### ✅ seaborn 
##### ✅ pylint 
##### ✅ autopep8

### To install all packages simply copy this command and paste it into your terminal 

``` 
pip install scikit-learn joblib pandas numpy matplotlib pytest seaborn pylint autopep8
```

## Files and data description

### There are 4 folders in our project
##### ✅ Data: 
  This folder includes the data required in this project which is the bank data CSV file.

##### ✅ Images: This folder includes two subfolders which are eda and results. 
✅ eda: This folder includes images related to the exploratory data analysis.
✅ results: This folder includes images related to the more advanced analysis.

✅ logs: This folder includes a churn_library.log file that tells us about what's happening in our function such as which function successfully runs and which function shows an error.

✅ models: This folder includes two Machine Learning models Random Forest Classifier and Logistic Regression which we saved in this folder to use the model again in the future. 
 
## Essential files of this project
  
### 1. churn_notebook.ipynb:
This is our notebook file which includes the algorithm to solve our business problem which is predicting customer churn this is the file we used to create two other files which are churn_library.py and churn_script_logging_and_tests.py.

### 2. churn_library.py:
This file perform the same task done by the churn_notebook.ipynb file but in this file, the code is properly split into functions that are clean and well-documented, and this file will be tested by the churn_script_logging_and_tests.py file

### 3. churn_script_logging_and_tests.py: 
This file is used to test our code which is written in the churn_library.py file this file also generates a churn_library.log file that generates a log when a function is tested.
  
### 4. conftest.py:
when we pass variables to the function the variable got updated and sometimes we need the updated variable as an argument to the next function so to build a module that stores variables for our files that is accessible throughout the directory.  

## Running Files

We can run our churn_library.py file using python or ipython command 

  ``` 
  python churn_library.py 
  ```

Our main file is churn_library.py which is then tested by the churn_script_logging_and_tests.py
So to perform the test we have to run the churn_script_logging_and_tests.py file using the python or ipython command.

  ```
  python churn_script_logging_and_tests.py 
  ```
  or
  
  ```
  ipython churn_script_logging_and_tests.py 
  ```

After running this command this module takes some time to run the test on the churn_library.py module and all the images and models will be stored in the respective destination when the test finishes.
 
