import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
from sklearn.feature_extraction import text

stop = text.ENGLISH_STOP_WORDS.union(STOPWORDS)
operators = set(('latest','new','th','day','Forget','difficult','late','went','beautiful','says','big'))
s_words = set(stop).union(operators)
#print(s_words)
df = pd.read_csv('train_set.csv', sep='\t')

#====================================================================================================== #
#========================================= Category: Politics ========================================= #
#====================================================================================================== #
df_category = df[df["Category"] == "Politics"]
df_content = df_category[["Content"]]

wordcloud = WordCloud(stopwords=s_words,mask=imread('brain.jpg')).generate(str(df_content))

plt.figure(figsize=(19,10))
plt.imshow(wordcloud)
plt.axis('off')

plt.savefig('politics.png', facecolor='k', bbox_inches='tight')
#====================================================================================================== #



#====================================================================================================== #
#========================================= Category: Film ============================================= #
#====================================================================================================== #
df_category = df[df["Category"] == "Film"]
df_content = df_category[["Content"]]

wordcloud = WordCloud(stopwords=s_words,mask=imread('brain.jpg')).generate(str(df_content))

plt.figure(figsize=(19,10))
plt.imshow(wordcloud)
plt.axis('off')

plt.savefig('film.png', facecolor='k', bbox_inches='tight')
#====================================================================================================== #



#====================================================================================================== #
#========================================= Category: Football ========================================= #
#====================================================================================================== #
df_category = df[df["Category"] == "Football"]
df_content = df_category[["Content"]]

wordcloud = WordCloud(stopwords=s_words,mask=imread('brain.jpg')).generate(str(df_content))

plt.figure(figsize=(19,10))
plt.imshow(wordcloud)
plt.axis('off')

plt.savefig('football.png', facecolor='k', bbox_inches='tight')
#====================================================================================================== #



#====================================================================================================== #
#========================================= Category: Business ========================================= #
#====================================================================================================== #
df_category = df[df["Category"] == "Business"]
df_content = df_category[["Content"]]

wordcloud = WordCloud(stopwords=s_words,mask=imread('brain.jpg')).generate(str(df_content))

plt.figure(figsize=(19,10))
plt.imshow(wordcloud)
plt.axis('off')

plt.savefig('business.png', facecolor='k', bbox_inches='tight')
#====================================================================================================== #



#====================================================================================================== #
#========================================= Category: Technology ======================================= #
#====================================================================================================== #
df_category = df[df["Category"] == "Technology"]
df_content = df_category[["Content"]]

wordcloud = WordCloud(stopwords=s_words,mask=imread('brain.jpg')).generate(str(df_content))

plt.figure(figsize=(19,10))
plt.imshow(wordcloud)
plt.axis('off')

plt.savefig('technology.png', facecolor='k', bbox_inches='tight')
#====================================================================================================== #