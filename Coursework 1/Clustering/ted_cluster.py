from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from wordcloud import STOPWORDS
from nltk import cluster
import pandas as pd
import warnings
import csv

#========================================================================================================#
warnings.filterwarnings('ignore')

stop = text.ENGLISH_STOP_WORDS.union(STOPWORDS)
operators = set(('latest','new','th','day','Forget','difficult','late','went','beautiful','says','big'))
s_words = set(stop).union(operators)
#========================================================================================================#
#========================================== Data preprocessing ==========================================#
#========================================================================================================#
df = pd.read_csv('train_set.csv', sep='\t')
df_content_list = df.Content.tolist()


vectorizer = TfidfVectorizer(stop_words=s_words, 
                             use_idf=True, 
                             smooth_idf=True,
                             max_features=1000)

svd_model = TruncatedSVD(n_components=500,
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])

svd_matrix = svd_transformer.fit_transform(df_content_list)

#========================================================================================================#
#============================================== Clustering ==============================================#
#========================================================================================================#
clusterer = cluster.KMeansClusterer(5, distance = cluster.util.cosine_distance, repeats=25)
assigned_clusters = clusterer.cluster(svd_matrix, assign_clusters=True)

#========================================================================================================#
#====================================== Export results to csv file ======================================#
#========================================================================================================#
df_categories_list = df.Category.tolist()
counters = [[0 for x in range(5)] for y in range(5)] 

for i in range(len(assigned_clusters)):
    if(df_categories_list[i] == 'Politics'):
        counters[assigned_clusters[i]][0] += 1          
    elif(df_categories_list[i] == 'Business'):
        counters[assigned_clusters[i]][1] += 1 
    elif(df_categories_list[i] == 'Football'):
        counters[assigned_clusters[i]][2] += 1 
    elif(df_categories_list[i] == 'Film'):
        counters[assigned_clusters[i]][3] += 1  
    elif(df_categories_list[i] == 'Technology'):
        counters[assigned_clusters[i]][4] += 1  
  
csv_out = [['' for x in range(6)] for y in range(5)]    

for i in range(5):
    csv_out[i][0] = 'Cluster' + str(i+1)
    csv_out[i][1] = str(round(counters[i][0]/(assigned_clusters.count(i)), 2)) 
    csv_out[i][2] = str(round(counters[i][1]/(assigned_clusters.count(i)), 2))
    csv_out[i][3] = str(round(counters[i][2]/(assigned_clusters.count(i)), 2))
    csv_out[i][4] = str(round(counters[i][3]/(assigned_clusters.count(i)), 2))
    csv_out[i][5] = str(round(counters[i][4]/(assigned_clusters.count(i)), 2))
    
with open('clustering_KMeans.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    wr.writerow(['', 'Politics', 'Business', 'Football', 'Film', 'Technology'])
    wr.writerow(csv_out[0])
    wr.writerow(csv_out[1])
    wr.writerow(csv_out[2])
    wr.writerow(csv_out[3])
    wr.writerow(csv_out[4])
#========================================================================================================#