import warnings
import pandas as pd
import time
import csv
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier






warnings.filterwarnings("ignore") # Some depreciate warnings regarding scikit in online learning


features = ['hour','day','dow','bidid','device_id','user_id','format','bidfloor','support_type','support_id',
            'device_type','device_os','device_language',
            'device_model','verticals_0','verticals_1','verticals_2','vertical_3','ad_id','bid_price',
            'won_price']

featuresInit = ['timestamp','bidid','device_id','user_id','format','bidfloor','support_type','support_id',
            'device_type','device_os','device_language',
            'device_model','verticals_0','verticals_1','verticals_2','vertical_3','ad_id','bid_price',
            'won_price']





#
def process_data():
    # Load data
 train = pd.read_csv("data/train")
 test = pd.read_csv("data/test")



 #Cleaning Data
 for column in featuresInit: 
  train[column].replace('None', np.nan, inplace= True)
  test[column].replace('None', np.nan, inplace= True)
 
 train.loc[~(train['won_price'] > 0), 'won_price']=np.nan
 train.loc[~(train['bidfloor'] > 0), 'bidfloor']=np.nan
 train.loc[~(train['bid_price'] > 0), 'bid_price']=np.nan
 
 test.loc[~(test['won_price'] > 0), 'won_price']=np.nan
 test.loc[~(test['bidfloor'] > 0), 'bidfloor']=np.nan
 test.loc[~(test['bid_price'] > 0), 'bid_price']=np.nan


 #Drop missing values with they represent only 0.014% of the entire data
 train.dropna(inplace=True)
 test.dropna(inplace=True)
 

 # Pre-processing non-number values
 le = LabelEncoder()
 for col in ['bidid','device_id','user_id','format','support_type','device_os','device_language','device_model']:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


 # feature scaling
 #scaler = StandardScaler()
 #for col in ['bidfloor','support_id','verticals_0','verticals_1','verticals_2','vertical_3','ad_id','bid_price','won_price']:
    #scaler.fit(list(train[col])+list(test[col]))
    #train[col] = scaler.transform(train[col])
    #test[col] = le.transform(test[col])
    
    
 # Add new features:
 train['day'] = train['timestamp'].apply(lambda x: (x - x%10000)/1000000) # day
 train['dow'] = train['timestamp'].apply(lambda x: ((x - x%10000)/1000000)%7) # day of week
 train['hour'] = train['timestamp'].apply(lambda x: x%10000/100) # hour


 test['day'] = test['timestamp'].apply(lambda x: (x - x%10000)/1000000) # day
 test['dow'] = test['timestamp'].apply(lambda x: ((x - x%10000)/1000000)%7) # day of week
 test['hour'] = test['timestamp'].apply(lambda x: x%10000/100) # hour
 
 
 # Create training set with unbalanced data

 train_majority = train[train.clicked==0]
 train_minority = train[train.clicked==1]


 train_minority_upsampled = resample(train_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=724277,    # to match majority class
                                 random_state=123) # reproducible results

 train_upsampled = pd.concat([train_majority, train_minority_upsampled])
 #print(train_upsampled['clicked'].value_counts())



 y = train_upsampled['clicked']
 X=train_upsampled[features]
 test=test[features]

# y = train['clicked']
# X = train[features]
    
 return X, y, test



#
# SGD-BASED LOGISTIC REGRESSION ~20 sec. to train
#
def logistic_regression( load_model=False):
    start = time.time()
    if load_model == False:
        print("*  Logistic regression model training started...")
    
    # Create Training Set
    X,y , test=process_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 

    # Load model 
    if load_model == True:
        print('✔  Loading model from previous training...')
        l_reg_file = open('models/logistic_regression_model.sav', 'rb')
        sgd_log_reg_model = pickle.load(l_reg_file)
        predictions = sgd_log_reg_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, predictions)
        print("✔  ROC AUC score on test set: {0:.3f}".format(score))
        return 0

    # Create SGD Logistic Regression Classifier
    sgd_log_reg_model = SGDClassifier(loss='log', penalty=None, fit_intercept=True,
                                      n_iter=5, learning_rate='constant', eta0=0.01)

    # Train Classifier
    sgd_log_reg_model.fit(X_train, y_train)
    print('✔  Logistic regression model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Run model on test set
    predictions = sgd_log_reg_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    score = roc_auc_score(y_test, predictions)
    print("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save model
    l_reg_file = open('models/logistic_regression_model.sav', "wb")
    pickle.dump(sgd_log_reg_model, l_reg_file)
    l_reg_file.close()
    print('✔  Logistic regression model saved...')



#
# GradientBoostingClassifier ~20 sec. to train
#
def GradientBoosting( load_model=False):
    start = time.time()
    if load_model == False:
        print("*  GradientBoostingClassifier model training started...")
    
    # Create Training Set
    X,y ,test =process_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 

    # Load model 
    if load_model == True:
        print('✔  Loading model from previous training...')
        GradientBoosting_file = open('models/GradientBoostingClassifier_model.sav', 'rb')
        sgd_GradientBoostingClassifier_model = pickle.load(GradientBoosting_file)
        predictions = sgd_GradientBoostingClassifier_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, predictions)
        print("✔  ROC AUC score on test set: {0:.3f}".format(score))


        return 0

    # Create SGD GradientBoostingClassifier
    sgd_GradientBoostingClassifier_model = GradientBoostingClassifier()

    # Train Classifier
    sgd_GradientBoostingClassifier_model.fit(X_train, y_train)
    print('✔  Logistic regression model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Run model on test set
    predictions = sgd_GradientBoostingClassifier_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    score = roc_auc_score(y_test, predictions)
    print("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save model
    GradientBoosting_file = open('models/GradientBoostingClassifier_model.sav', "wb")
    pickle.dump(sgd_GradientBoostingClassifier_model, GradientBoosting_file)
    GradientBoosting_file.close()
    print('✔  GradientBoostingClassifier_model saved...')

#
# RANDOM FOREST ~ 20 min to train
#
def random_forest(load_model=False, write=False):
    start = time.time()
    if load_model == False:
        print("*  Random forest model training started...")

    # Create training set
    X,y ,test =process_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 


    # Load model instead of training again..
    if load_model == True:
        print('✔  Loading model from previous training...')
        r_forest_file = open('models/random_forest_model.sav', 'rb')
        random_forest_final = pickle.load(r_forest_file)
        
        # Run model on test set
        predictions = random_forest_final.predict_proba(X_test)[:, 1]
        
        
        # Evaluate model
        
        #print ('Log Loss:')
        #print (log_loss(y_test, predictions))
        #print ('RMSE:')
        #print (mean_squared_error(y_test, np.compress([False, True], predictions, axis=1))**0.5) # RMSE
        score = roc_auc_score(y_test, predictions)
        print('✔  ROC AUC score on test set: {0:.3f}'.format(score))
        print('✔ F1_score:' ,f1_score(y_test, random_forest_final.predict(X_test), average='weighted') )
        print('✔ Confusion_matrix :', confusion_matrix(y_test, random_forest_final.predict(X_test) ))  
        print(classification_report(y_test, random_forest_final.predict(X_test) )) 
        print("✔ Accuracy :", accuracy_score(y_test, random_forest_final.predict(X_test)))

        
        

        r_forest_file.close()
        
        
        # Compute ROC curve and ROC area 
        y_pred_proba = random_forest_final.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('random_forest Classifier ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr,tpr,color='darkorange',label='random_forest ROC area = %0.4f)' % auc)
        plt.legend(loc=4)
        plt.show()
       
        if write==True:
         #Predict results for Test
         result=random_forest_final.predict(test)
         test1 = pd.read_csv("data/test")
          #Cleaning Data
         for column in featuresInit: 
          test1[column].replace('None', np.nan, inplace= True)
 
         test1.loc[~(test1['won_price'] > 0), 'won_price']=np.nan
         test1.loc[~(test1['bidfloor'] > 0), 'bidfloor']=np.nan
         test1.loc[~(test1['bid_price'] > 0), 'bid_price']=np.nan


 #Drop missing values with they represent only 0.014% of the entire data
         test1.dropna(inplace=True)
         
         
         test1['clicked'] = result
         test1.to_csv('data/results.csv')
        
        return 0
    
    # Train random forest classifier
    params = {'max_depth': [3, 10, None]}
    random_forest_model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30,
                                                 n_jobs=-1)
    grid_search = GridSearchCV(random_forest_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    print('✔  Random forest model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use best paramter for final model
    random_forest_final = grid_search.best_estimator_

    # Evaluate model
    predictions = random_forest_final.predict_proba(X_test)[:, 1]
    
        
    #print ('Log Loss:')
    #print (log_loss(y_test, predictions))
    #print ('RMSE:')
    #print (mean_squared_error(y_test, np.compress([False, True], predictions, axis=1))**0.5) # RMSE
    score = roc_auc_score(y_test, predictions)
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score))
    print('✔ F1_score:' ,f1_score(y_test, random_forest_final.predict(X_test), average='weighted') )
    print('✔ Confusion_matrix :', confusion_matrix(y_test, random_forest_final.predict(X_test) ))  
    print(classification_report(y_test, random_forest_final.predict(X_test) )) 
    print("✔ Accuracy :", accuracy_score(y_test, random_forest_final.predict(X_test)))

    r_forest_file.close()
        
        
    

    # Save Model
    random_forest_file = open('models/random_forest_model.sav', "wb")
    pickle.dump(random_forest_final, random_forest_file)
    random_forest_file.close()
    print('✔  Random forest model saved...')
    return 0

#
# MAIN
#
def main():
    
    # Logistic Regression
    #print('SGD Based Logistic Regression')
    #logistic_regression(load_model=True)
    #print("✔  Done")
    
    #GradientBoostingClassifier
    #print('GradientBoostingClassifier')
    #GradientBoosting(load_model=True)
    #print("✔  Done")


    #random_forest
    print('Random Forest')
    random_forest(load_model=True, write=True)
    print("✔  Done")
    

if __name__ == '__main__':
    main()
    