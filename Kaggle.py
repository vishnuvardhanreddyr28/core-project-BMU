# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import quantreg
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class osic():
    def __init__(self):
        self.train=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
        self.test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
        self.sub   = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )
        print('shape of train data ',self.train.shape)
        print('shape of test data ',self.test.shape)
        print('shape of sub data ',self.sub.shape)
        print(' ')
    
    def preprocess_data(self):
        columns=['SmokingStatus','Sex']
        for fitcolumns in columns:
            le = preprocessing.LabelEncoder()
            le.fit(self.train[fitcolumns])
            self.train[fitcolumns]=le.transform(self.train[fitcolumns])
        
        
        print('Perform minmax scaling ')
        print(' ')
        columns=['Percent','Age','Sex','SmokingStatus']
        for transformcolumns in columns:
            scaler = MinMaxScaler()
            scaler.fit(np.array(self.train[transformcolumns]).reshape(-1,1))
            self.train[transformcolumns]=scaler.transform(np.array(self.train[transformcolumns]).reshape(-1,1)) 
            
            
    def train_model(self):
        y_train=self.train['FVC']
        x_train=self.train[['Weeks','Percent','Age','Sex','SmokingStatus']]
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
        print('Linear regression ')
        reg = LinearRegression().fit(x_train,y_train)
        print('Accuracy on the test data set ')
        print(reg.score(x_test,y_test))
        
        
        
        
    def kaggle(self):
        self.train=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
        self.test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
        self.sub   = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )
        self.train['traintest'] = 0
        self.test ['traintest'] = 1
        
        #Extract weeks from the submission file 
        
        self.sub['Weeks']   = self.sub['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )
        self.sub['Patient'] = self.sub['Patient_Week'].apply( lambda x: x.split('_')[0] ) 
        
        #All patientid in test file are same as the submission file 
        #combine test and train data
        
        self.traintest = pd.concat( (self.train,self.test) )
        self.traintest.sort_values( ['Patient','Weeks'], inplace=True )
        print(self.traintest.shape)
        
        columns=['SmokingStatus','Sex']
        for fitcolumns in columns:
            le = preprocessing.LabelEncoder()
            le.fit(self.traintest[fitcolumns])
            self.traintest[fitcolumns]=le.transform(self.traintest[fitcolumns])
        
        
        print('Perform minmax scaling ')
        print(' ')
        columns=['Percent','Age','Sex','SmokingStatus']
        for transformcolumns in columns:
            scaler = MinMaxScaler()
            scaler.fit(np.array(self.traintest[transformcolumns]).reshape(-1,1))
            self.traintest[transformcolumns]=scaler.transform(np.array(self.traintest[transformcolumns]).reshape(-1,1)) 
        
        #Kaggle has provided a unique metric for calculation of the score which takes in the prediciton score as well as
        #the confidence value of each prediction.
        #More details of the scoring mechanism can be found on kaggle 
    
        
        modelL = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', self.traintest).fit( q=0.15 )
        model  = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', self.traintest).fit( q=0.50 )
        modelH = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', self.traintest).fit( q=0.85 )
        print(model.summary())
        print(' ')
        
        
        
        self.traintest['ypredL'] = modelL.predict( self.traintest ).values
        self.traintest['ypred']  = model.predict( self.traintest ).values
        self.traintest['ypredH'] = modelH.predict( self.traintest ).values
        self.traintest['ypredstd'] = 0.5*np.abs(self.traintest['ypredH'] - self.traintest['ypred'])+0.5*np.abs(self.traintest['ypred'] - self.traintest['ypredL'])
        
        
        
        
        
        
        df = self.traintest.loc[self.traintest.traintest==1 ,['Patient','Percent','Age','Sex','SmokingStatus']]
        self.test = pd.merge( self.sub, df, on='Patient', how='left' )
        self.test.sort_values( ['Patient','Weeks'], inplace=True )
        
        self.test['ypredL'] = modelL.predict( self.test ).values
        self.test['FVC']    = model.predict( self.test ).values
        self.test['ypredH'] = modelH.predict( self.test ).values
        self.test['Confidence'] = np.abs(self.test['ypredH'] - self.test['ypredL']) / 2

        self.test[['Patient_Week','FVC','Confidence']].to_csv('submission.csv', index=False)
        
        
        
        

        
        

        
        
        
if __name__=='__main__':
    osicobj=osic()
    osicobj.preprocess_data()
    osicobj.train_model()
    osicobj.kaggle()
        
        
                


        
        
        
            
            
        
        
            
        
        

        
            
        
        
        
    
        
        
        
    
        
    
        
        