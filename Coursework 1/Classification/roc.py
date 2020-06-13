from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD  
from sklearn.feature_extraction import text
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



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
df_categories_list = df.Category

category_names = ['Politics', 'Business', 'Football', 'Film', 'Technology']


vectorizer = TfidfVectorizer(stop_words=s_words, use_idf=True, smooth_idf=True, max_features=1000)

tfidf_content = vectorizer.fit_transform(df_content_list)


svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10, random_state=42)

lsi_content = svd_model.fit_transform(tfidf_content)


df_categories_binarize = label_binarize(df_categories_list, classes=category_names)
n_classes = df_categories_binarize.shape[1]


X_train, X_test, Y_train, Y_test = train_test_split(lsi_content, df_categories_list, test_size=.5, random_state=0)


clf_roc = ['NB_roc_plot', 'RF_roc_plot', 'SVM_roc_plot', 'KNN_roc_plot']

title = ['NB ROC Plot', 'RF ROC Plot', 'SVM ROC Plot', 'KNN ROC Plot']


for i, roc, title in zip(range(4), clf_roc, title):
    
    if (i == 0):
        yScore = cross_val_predict(OneVsRestClassifier(BernoulliNB()), lsi_content, df_categories_binarize, cv=5)        
    elif (i == 1):
        yScore = cross_val_predict(OneVsRestClassifier(RandomForestClassifier()), lsi_content, df_categories_binarize, cv=5)        
    elif(i == 2):
        yScore = cross_val_predict(OneVsRestClassifier(SVC(kernel='linear', probability=True)), lsi_content, df_categories_binarize, cv=5)        
    else:
        yPred = KNN_Classifier(3, X_train, X_test, Y_train)  
        
        yScore = label_binarize(yPred, classes=category_names)
        
        df_categories_binarize = label_binarize(Y_test, classes=category_names)
        
   
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(df_categories_binarize[:, i], yScore[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(df_categories_binarize.ravel(), yScore.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
    
    colors = ['red', 'orange', 'blue', 'pink', 'green']
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of {0} (area = {1:0.2f})'.format(category_names[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    plt.savefig(roc)