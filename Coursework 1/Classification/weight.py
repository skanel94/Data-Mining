from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold
from wordcloud import STOPWORDS
from sklearn.svm import SVC
import pandas as pd


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


weights = list(range(10))


metrics_accuracy = [0 for j in range(len(weights))] 
metrics_precission = [0 for j in range(len(weights))] 

for weight, i in zip(weights, range(len(weights))):

    print(i)

    df_weighted_list = ['' for _ in range(len(df_content_list))]

    for k in range(len(df_content_list)):
        df_weighted_list[k] = df_content_list[k] + weight * df_title_list[k]
    
    
   
    vectorizer = TfidfVectorizer(stop_words=s_words, max_features=1000)
    
    
    tfidf_content = vectorizer.fit_transform(df_weighted_list)
    
    
    svd_model = TruncatedSVD(n_components=100, algorithm='randomized')
    
    
    lsi_weighted = svd_model.fit_transform(tfidf_content)
    
    #========================================================================================================#
    #======================================= 10-Fold Cross Validation =======================================#
    #========================================================================================================#
    
    kf = KFold(n_splits=5, shuffle=True)
    
    counter = 0
    
    for train_index, test_index in kf.split(df_content_list):
        X_train = lsi_weighted[train_index]
        X_test  = lsi_weighted[test_index]
        
        Y_train = df_categories_list[train_index]
        Y_test  = df_categories_list[test_index]



        classifier_SVM = SVC(kernel='linear').fit(X_train, Y_train)
        
     
        yPred_SVM = classifier_SVM.predict(X_test)
        
        counter += accuracy_score(Y_test, yPred_SVM)
       
        

    metrics_accuracy[i] = counter/5
     
           
           
        

plt.figure(figsize=(12, 8))
plt.plot(weights, metrics_accuracy)
plt.title('Accuracy - Title weight')
plt.xlabel('Title weight')
plt.ylabel('Prediction Accuracy')
plt.grid(True)

plt.savefig('accuracy_weight.png')