"""
In this module we write important function that we 
are going to test

Author: Ashbab khan
Date: 15-Jan-2023
"""

# library doc string
import os
import joblib
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'

# list of the numerical column that we need to make a ML model
keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn',"Churn"]
 
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
    # making a column churn which store 0 or 1 
    # 0 for Existing_Customer and 1 for other 
    df["Churn"] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # plotting histogram 
    df['Churn'].hist()

    # saving these plot image into image/eda path
    plt.savefig("./images/eda/churn_distribution.png")

    # making plot of distribution of the customer age column 
    df['Customer_Age'].hist();
    # then saving it to image/eda path
    plt.savefig("./images/eda/Customer_age_distribution.png")
    
    # making a bar chart of marital status column
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    # saving this bar plot to image/eda path
    plt.savefig("./images/eda/marital_status_distribution.png")

    # making a histplot of Total_transaction_Ct column
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    # saving this histplot to the path images/eda
    plt.savefig("./images/eda/total_transaction_distribution.png")
    
    # showing correlation between our columns using heatmap
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    # saving our heatmap to image/eda path
    plt.savefig("./images/eda/heatmap.png")
    plt.show()
    return df

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
        gender_groups = df.groupby(lists).mean(numeric_only=True)['Churn']

        for val in df[lists]:
            list_values.append(gender_groups.loc[val])

        names = lists+"_Churn"
        df[names] = list_values    

    return df[keep_cols]

def perform_feature_engineering(df):
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
    df1 = df.head(100)
    # deleting output column making a input features X
    X = df1.drop("Churn",axis=1)
    # making our target variable
    y = df1["Churn"]

    # splitting our input features X and target variable y
    # into four variables and distributing 30% to test variable
    # and 70% to training variables
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=42)
    print("-------------Success part1------------")

    # Initializing Random Forest Classifier algorithm    
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    # initializing our Logistic model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    print("-------------Success part2------------")

    # making a list of different hyper parameters and their
    # different value used in GridSearchCV
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    # GridSearchCV this will used to select best value of hyper
    # parameters given in the param_grid dictionary 
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    print("-------------Success part 3-I------------")

    # fitting training data to our Random Classifier Model
    cv_rfc.fit(X_train, y_train)
    print("-------------Success part 3-II------------")
    # fitting training data to out Logistic Regression model 
    lrc.fit(X_train, y_train)
    print("-------------Success part4------------")

    # predicting the X_train and X_test output using the best 
    # hyper parameter values in Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predicting the X_train and X_test output in
    # Logistic Regression 
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    print("-------------Success part5------------")

    # Saving our important variable to confest.py so 
    # we can easily access these variable in future
    pytest.lrc = lrc
    pytest.cv_rfc = cv_rfc
    pytest.y_train_preds_lr = y_train_preds_lr
    pytest.y_train_preds_rf = y_train_preds_rf
    pytest.y_test_preds_lr = y_test_preds_lr
    pytest.y_test_preds_rf = y_test_preds_rf 
    print("------------------------return successful----------------------------")
    return X_train,X_test,y_train,y_test

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
    
    # Plotting and saving classification report of Random Forest model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/random_forest_results.png")

    # Plotting and saving classification report of Logistic Regression model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/logistic_regression.png")

def feature_importance_plot(cv_rfc, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Saving our plot in the output_pth
    plt.savefig(output_pth)

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
    # getting these important variable cv_rfc and lrc from
    # confest.py file
    cv_rfc = pytest.cv_rfc
    lrc = pytest.lrc
    # saving our model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # loading the rfc_model (Random Forest Classifier)
    rfc_model = joblib.load('./models/rfc_model.pkl')
    # predicting probability of getting 1 and 0 usinf RFC model
    y_test_preds_proba_rf = rfc_model.predict_proba(X_test)

    # loading the lr_model (Logistic Regression)
    lr_model = joblib.load('./models/logistic_model.pkl')
    y_test_preds_proba_lr = lr_model.predict_proba(X_test)

    # plotting and saving roc_curve of Logistic Regression model
    fpr_lr,tpr_lr,thr_lr = roc_curve(y_test,y_test_preds_proba_lr[:,1])
    plt.figure(figsize=(15, 8))
    # ax = plt.gca()
    plt.plot(fpr_lr,tpr_lr,color="blue",label="Logistic regression")
    plt.plot([0,1],[0,1])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")

    
    # rfc_disp = roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    # lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/logistic_roc_plot.png")

    # plotting and saving roc_curve of Random Forest Classifier model 
    fpr_rfc,tpr_rfc,thr_rfc = roc_curve(y_test,y_test_preds_proba_rf[:,1])
    plt.figure(figsize=(15, 8))
    # ax = plt.gca()
    plt.plot(fpr_rfc,tpr_rfc,color="green",label="Random forest classifier")
    plt.plot([0,1],[0,1])
    plt.legend()
    plt.savefig("./images/results/rfc_roc_curve.png")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()
