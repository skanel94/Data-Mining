from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
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
df_categories_list = df.Category

category_names = ['Politics', 'Business', 'Football', 'Film', 'Technology']

components = [1, 10, 25, 50, 100, 150, 200]

accuracy_comp = [0 for j in range(len(components))]

vectorizer = TfidfVectorizer(stop_words=s_words, use_idf=True, smooth_idf=True, max_features=1000)


tfidf_content = vectorizer.fit_transform(df_content_list)


for comp, i in zip(components, range(len(components))):

    svd_model = TruncatedSVD(n_components=comp, algorithm='randomized', n_iter=10, random_state=42)

    lsi_content = svd_model.fit_transform(tfidf_content)
   
    counter = 0
    
    kf = KFold(n_splits=5)
   
    for train_index, test_index in kf.split(df_content_list):

        X_train = lsi_content[train_index]
        X_test  = lsi_content[test_index]

        Y_train = df_categories_list[train_index]
        Y_test  = df_categories_list[test_index]        
        
        classifier_NB = SVC(kernel='linear').fit(X_train, Y_train)

        yPred_NB = classifier_NB.predict(X_test)
              
        counter += accuracy_score(Y_test, yPred_NB)
       
       
    accuracy_comp[i] = counter/5  


plt.figure(figsize=(12, 8))
plt.plot(components, accuracy_comp)
plt.title('Accuracy - Components')
plt.xlabel('No of Components')
plt.ylabel('Prediction Accuracy')
plt.grid(True)

plt.savefig('accuracy_components.png')