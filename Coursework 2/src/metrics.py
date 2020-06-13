from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd
import csv
import numpy as np



df = pd.read_csv('train.tsv', sep='\t')

data = list()

for attr in df.columns[:-2]:
    if (type(df[attr][0]) is str):    
        values = (df[attr]).value_counts().sort_index().index
        
        encoder = LabelEncoder().fit(values)
        encoded_col = encoder.transform(df[attr]).tolist()        
    else:
        encoded_col = df[attr].tolist()

    data.append(encoded_col)    


encoded_data = np.transpose(np.array(data))

#========================================================================================================#
#======================================= 10-Fold Cross Validation =======================================#
#========================================================================================================#
metrics = [0 for i in range(3)]

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(df.Id):
    X_train = encoded_data[train_index]
    X_test  = encoded_data[test_index]
    
    Y_train = df.Label[train_index]
    Y_test = df.Label[test_index]
    
#--------------------------------------------------------------------------------------------------------#
#--------------------------------- Naive Bayes Classifier ( Benroulli ) ---------------------------------#
#--------------------------------------------------------------------------------------------------------#    
    classifier_NB = GaussianNB().fit(X_train, Y_train)

    yPred_NB = classifier_NB.predict(X_test)

    metrics[0] += accuracy_score(Y_test, yPred_NB)

#--------------------------------------------------------------------------------------------------------#
#--------------------------------------- Random Forest Classifier ---------------------------------------#
#--------------------------------------------------------------------------------------------------------#     
    classifier_RF = RandomForestClassifier().fit(X_train, Y_train)

    yPred_RF = classifier_RF.predict(X_test)
    
    metrics[1] += accuracy_score(Y_test, yPred_RF)
  
#--------------------------------------------------------------------------------------------------------#
#------------------------------------- Support Vector Machines (SVC) ------------------------------------#
#--------------------------------------------------------------------------------------------------------#  
    classifier_SVM = SVC().fit(X_train, Y_train)

    yPred_SVM = classifier_SVM.predict(X_test)
       
    metrics[2] += accuracy_score(Y_test, yPred_SVM)
      
#--------------------------------------------------------------------------------------------------------#        
#========================================================================================================#
#====================================== Export results to csv file ======================================#
#========================================================================================================#  
csv_out = [str('Accuracy'), str(metrics[0]/10), str(metrics[1]/10), str(metrics[2]/10)]

with open('EvaluationMetric_10fold.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'])
    wr.writerow(csv_out)
#========================================================================================================#