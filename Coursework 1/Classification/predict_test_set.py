from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  
from sklearn.feature_extraction import text
from wordcloud import STOPWORDS
from sklearn.svm import SVC
import pandas as pd
import csv



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


df_test = pd.read_csv('test_set.csv', sep='\t')

df_test_content_list = df_test.Content.tolist()
df_test_title_list = df_test.Title.tolist()
df_test_id_list = df_test.Id.tolist()



df_weighted_list = ['' for _ in range(len(df_content_list))]

for i in range(len(df_content_list)):
    df_weighted_list[i] = df_content_list[i] + 2 * df_title_list[i]
 
    
    
df_test_weighted_list = ['' for _ in range(len(df_test_content_list))]

for i in range(len(df_test_content_list)):
    df_test_weighted_list[i] = df_test_content_list[i] + 2 * df_test_title_list[i]



vectorizer = TfidfVectorizer(stop_words=s_words, use_idf=True, smooth_idf=True, max_features=1000)


tfidf_content = vectorizer.fit_transform(df_weighted_list)

tfidf_test_content = vectorizer.fit_transform(df_test_weighted_list)


svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10, random_state=42)


lsi_weighted = svd_model.fit_transform(tfidf_content)

lsi_weighted_test = svd_model.fit_transform(tfidf_test_content)


#--------------------------------------------------------------------------------------------------------#
#------------------------------------- Support Vector Machines (SVC) ------------------------------------#
#--------------------------------------------------------------------------------------------------------#  
classifier_SVM = SVC(kernel='linear').fit(lsi_weighted, df_categories_list)

yPred_SVM = classifier_SVM.predict(lsi_weighted_test)





#========================================================================================================#
#====================================== Export results to csv file ======================================#
#========================================================================================================#         
csv_out = [['' for x in range(2)] for y in range(len(yPred_SVM))] 

for i in range(len(yPred_SVM)):       
    csv_out[i][0] = df_test_id_list[i]
    csv_out[i][1] = yPred_SVM[i]

 
with open('testSet_categories.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(['ID', 'Predicted_Category'])
    for i in range(len(yPred_SVM)):       
        wr.writerow(csv_out[i])


#========================================================================================================#
