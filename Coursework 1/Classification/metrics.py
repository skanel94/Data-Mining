from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD  
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold
from wordcloud import STOPWORDS
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import csv



def KNN_Classifier(k, X_train, X_test, Y_train):
    predicted = []
    
    for x in range(int(len(X_test))):
        distances = []
        neighbors = []
        votes = {}
        
        
        dist = euclidean_distances(X_train, X_test[x].reshape(1,-1))
        
        [distances.append(d[0]) for d in sorted(enumerate(dist), key=lambda x:x[1])]
 
                
        for i in range(k):
            neighbors.append(np.array(Y_train)[distances[i]])
        
        
        for i in range(len(neighbors)):      
            
            if neighbors[i] not in votes:
                votes[neighbors[i]] = 1
            else:
                votes[neighbors[i]] += 1
                
        category = max(votes, key=votes.get)
        
        predicted.append(category)
    
    return predicted


#========================================================================================================#
stop = text.ENGLISH_STOP_WORDS.union(STOPWORDS)
operators = set(('latest','new','th','day','Forget','difficult','late','went','beautiful','says','big'))
s_words = set(stop).union(operators)
#========================================================================================================#
#========================================== Data preprocessing ==========================================#
#========================================================================================================#
df = pd.read_csv('train_set.csv', sep='\t')

df_content_list = df.Content.tolist()
df_title_list = df.Title.tolist()
df_categories_list = df.Category

category_names = ['Politics', 'Business', 'Football', 'Film', 'Technology']


df_weighted_list = ['' for _ in range(len(df_content_list))]

for i in range(len(df_content_list)):
    df_weighted_list[i] = df_content_list[i] + 2 * df_title_list[i]


vectorizer = TfidfVectorizer(stop_words=s_words, use_idf=True, smooth_idf=True, max_features=1000)

tfidf_content = vectorizer.fit_transform(df_weighted_list)


svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10, random_state=42)

lsi_weighted = svd_model.fit_transform(tfidf_content)

#========================================================================================================#
#======================================= 10-Fold Cross Validation =======================================#
#========================================================================================================#
metrics = [[0 for j in range(4)] for i in range(5)] 

kf = KFold(n_splits=10, shuffle=True)


for train_index, test_index in kf.split(df_content_list):
    X_train = lsi_weighted[train_index]
    X_test  = lsi_weighted[test_index]
    
    Y_train = df_categories_list[train_index]
    Y_test  = df_categories_list[test_index]
    
    Y_test_binarize = label_binarize(Y_test, classes=category_names)

#--------------------------------------------------------------------------------------------------------#
#--------------------------------- Naive Bayes Classifier ( Benroulli ) ---------------------------------#
#--------------------------------------------------------------------------------------------------------#    
    classifier_NB = BernoulliNB().fit(X_train, Y_train)


    yPred_NB = classifier_NB.predict(X_test)
    
    
    yPred_NB_binarize = label_binarize(yPred_NB, classes=category_names)
 
    metrics[0][0] += accuracy_score(Y_test, yPred_NB)
    metrics[1][0] += precision_score(Y_test, yPred_NB, average='weighted')
    metrics[2][0] += recall_score(Y_test, yPred_NB, average='weighted')
    metrics[3][0] += f1_score(Y_test, yPred_NB, average='weighted')      
    metrics[4][0] += roc_auc_score(Y_test_binarize, yPred_NB_binarize, average='weighted')  
    
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------- Random Forest Classifier ---------------------------------------#
#--------------------------------------------------------------------------------------------------------#     
    classifier_RF = RandomForestClassifier().fit(X_train, Y_train)


    yPred_RF = classifier_RF.predict(X_test)
    
    
    yPred_RF_binarize = label_binarize(yPred_RF, classes=category_names)
    
    metrics[0][1] += accuracy_score(Y_test, yPred_RF)
    metrics[1][1] += precision_score(Y_test, yPred_RF, average='weighted')
    metrics[2][1] += recall_score(Y_test, yPred_RF, average='weighted')
    metrics[3][1] += f1_score(Y_test, yPred_RF, average='weighted')   
    metrics[4][1] += roc_auc_score(Y_test_binarize, yPred_RF_binarize, average='weighted') 
     
#--------------------------------------------------------------------------------------------------------#
#------------------------------------- Support Vector Machines (SVC) ------------------------------------#
#--------------------------------------------------------------------------------------------------------#  
    classifier_SVM = SVC(C=1.0, kernel='linear', gamma='auto', probability=True).fit(X_train, Y_train)
    
 
    yPred_SVM = classifier_SVM.predict(X_test)
    
   
    yPred_SVM_binarize = label_binarize(yPred_SVM, classes=category_names)
    
    metrics[0][2] += accuracy_score(Y_test, yPred_SVM)
    metrics[1][2] += precision_score(Y_test, yPred_SVM, average='weighted')
    metrics[2][2] += recall_score(Y_test, yPred_SVM, average='weighted')
    metrics[3][2] += f1_score(Y_test, yPred_SVM, average='weighted')   
    metrics[4][2] += roc_auc_score(Y_test_binarize, yPred_SVM_binarize, average='weighted') 
       
#--------------------------------------------------------------------------------------------------------#
#------------------------------------------ K-Nearest Neighbors------------------------------------------#
#--------------------------------------------------------------------------------------------------------#  
    yPred_KNN = KNN_Classifier(3, X_train, X_test, Y_train)  
        
    
    yPred_KNN_binarize = label_binarize(yPred_KNN, classes=category_names)    
           
    metrics[0][3] += accuracy_score(Y_test, yPred_KNN)
    metrics[1][3] += precision_score(Y_test, yPred_KNN, average='weighted')
    metrics[2][3] += recall_score(Y_test, yPred_KNN, average='weighted')
    metrics[3][3] += f1_score(Y_test, yPred_KNN, average='weighted')  
    metrics[4][3] += roc_auc_score(Y_test_binarize, yPred_KNN_binarize, average='weighted') 
        
#========================================================================================================#
#====================================== Export results to csv file ======================================#
#========================================================================================================#       
csv_out = [['' for x in range(5)] for y in range(5)]    

metrics_name = ['Accuracy', 'Precision', 'Recall', 'F-Measure', 'AUC']

for i in range(5):
    csv_out[i][0] = metrics_name[i]
    csv_out[i][1] = str(metrics[i][0]/10)
    csv_out[i][2] = str(metrics[i][1]/10)
    csv_out[i][3] = str(metrics[i][2]/10)
    csv_out[i][4] = str(metrics[i][3]/10)
    
with open('EvaluationMetric_10fold.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN'])
    wr.writerow(csv_out[0])
    wr.writerow(csv_out[1])
    wr.writerow(csv_out[2])
    wr.writerow(csv_out[3])
    wr.writerow(csv_out[4])
#========================================================================================================#