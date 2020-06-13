from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv


df_train = pd.read_csv('train.tsv', sep='\t')
df_test = pd.read_csv('test.tsv', sep='\t')


encoded_data_train = list()
encoded_data_test = list()

for attr in df_train.columns[:-2]:
    if (type(df_train[attr][0]) is str):    
        values = (df_train[attr]).value_counts().sort_index().index
        encoder = LabelEncoder().fit(values)
        
        encoded_col_train = encoder.transform(df_train[attr]).tolist()  
        
        encoded_col_test = encoder.transform(df_test[attr]).tolist()        
    else:  
        if len(list(set(df_train[attr]))) > 5:      
            encoded_col_train = pd.cut(df_train[attr], bins = 5, labels=[0,1,2,3,4]).tolist() 
            
            encoded_col_test = pd.cut(df_test[attr], bins = 5, labels=[0,1,2,3,4]).tolist() 
        else:
            encoded_col_train = df_train[attr]
            encoded_col_test = df_test[attr]
            
            
    encoded_data_train.append(encoded_col_train)
    encoded_data_test.append(encoded_col_test)


data_train = list()
data_test = list()

attributes = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 14, 19]

for attr in attributes:        
    data_train.append(encoded_data_train[attr]) 
    data_test.append(encoded_data_test[attr]) 


data_train = np.transpose(np.array(data_train))
data_test = np.transpose(np.array(data_test))

#--------------------------------------------------------------------------------------------------------#
#------------------------------------- Support Vector Machines (SVC) ------------------------------------#
#--------------------------------------------------------------------------------------------------------#  
classifier_RF = RandomForestClassifier().fit(data_train, df_train.Label)
    
yPred_RF = classifier_RF.predict(data_test)


with open('testSet_Predictions.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(["Client_ID", "Predicted_Label"])
    for test_id, predict in zip(df_test.Id, yPred_RF):
        if predict == 1:           
            csv_out = [str(test_id), "Good"]
        else:
            csv_out = [str(test_id), "Bad"]
        wr.writerow(csv_out)  
#========================================================================================================#