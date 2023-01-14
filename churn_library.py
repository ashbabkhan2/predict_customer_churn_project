"""
In this module we write important function that solve our business problem

Author: Ashbab khan
Date: 12-Jan-2023
"""

# library doc string


# import libraries
# import os
# export 
# DISPLAY = unix$DISPLAY
import os
# os.environ['QT_QPA_PLATFORM']='offscreen'
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'


# global_df = import_data("./data/bank_data.csv") 
def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    return df
#     pass


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
#     # checking the rows and columns in the dataframe
#     df.shape
#     # checking is there any NULL values in our columns
#     df.isnull().sum()
#     # seeing basic statistics of our numerical column
#     df.describe()
#     # converting the categorical value in Attrition_Flag column into binary value (0        or 1) and the storing in new column (Churn)
    df["Churn"] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    
    # plotting a histogram of distribution of our churn column
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    # saving these plot image into image/eda path
    plt.savefig("./images/eda/churn_distribution.png")

    # making plot of distribution of the customer age column
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    # then saving it to image/eda path
    plt.savefig("./images/eda/Customer_age_distribution.png")
    
    # making a bar chart of marital status column
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    # saving this bar plot to image/eda path
    plt.savefig("./images/eda/marital_status_distribution.png")

    # making a histplot of Total_transaction_Ct column
    plt.figure(figsize=(20,10)) 
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    # saving this histplot to the path images/eda
    plt.savefig("./images/eda/total_transaction_distribution.png")
    
    # showing correlation between our columns using heatmap
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    # saving our heatmap to image/eda path
    plt.savefig("./images/eda/heatmap.png")
    plt.show()
    return df
#     pass



def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for lists in category_lst:
        list_values=[]
        gender_groups = df.groupby(lists).mean()['Churn']

        for val in df[lists]:
            list_values.append(gender_groups.loc[val])

        names = lists+"Churn"
        df[names] = list_values    

    print(df.columns)    
    return df


#     pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df.drop("Churn",axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=42)

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # print('random forest results')
    # print('test results')
    # print(classification_report(y_test, y_test_preds_rf))
    # print('train results')
    # print(classification_report(y_train, y_train_preds_rf))
    # print('logistic regression results')
    # print('test results')
    # print(classification_report(y_test, y_test_preds_lr))
    # print('train results')
    # print(classification_report(y_train, y_train_preds_lr))

    return [y_train,y_test,y_train_preds_lr,y_train_preds_rf,y_test_preds_lr,y_test_preds_rf]


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
#     pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
#     pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
#     pass