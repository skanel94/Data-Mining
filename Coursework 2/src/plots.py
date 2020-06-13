import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('train.tsv', sep='\t')

index_good = list()
index_bad = list()

for i in range(df.shape[0]):
    if (df.Label[i] == 1):
        index_good.append(i)
    else:
        index_bad.append(i)
     
        
   
for attr in df.columns[:-2]:
    if (type(df[attr][0]) is str):
        good = (df[attr])[index_good].value_counts().sort_index()
        bad = (df[attr])[index_bad].value_counts().sort_index()
        
        values = (df[attr]).value_counts().sort_index().index
                                           
        N = len(values)
        ind = np.arange(N) 
        width = 0.3     
        
        fig = plt.figure(figsize=(11, 7))       
        ax = fig.add_subplot(111)
                
        rects1 = ax.bar(ind, good, width, color='g')
        rects2 = ax.bar(ind + width, bad, width, color='r')
        
        ax.set_title(attr)
        ax.set_ylabel('# Customers')
        ax.set_xticks(ind + width/2)
        ax.set_xticklabels(values)
        ax.legend( (rects1[0], rects2[0]), ('Good', 'Bad'), loc = 2, fontsize = 18)
        
        plt.show()
        fig.savefig(attr)
    else:
        good = (df[attr])[index_good]
        bad = (df[attr])[index_bad]
       
        fig = plt.figure(figsize=(11, 7))  
        ax = fig.add_subplot(111)
        ax.set_title(attr)
        
        box = plt.boxplot([good, bad],  widths = 0.5, patch_artist=True)
        for patch, color in zip(box['boxes'], ['green', 'red']):
            patch.set_facecolor(color)
        
        
        hB = plt.plot([1,1],'g')
        hR = plt.plot([1,1],'r')
        ax.legend((hB[0], hR[0]),('Good', 'Bad'), loc = 9, fontsize = 18)
        hB[0].set_visible(False)
        hR[0].set_visible(False)
        
        plt.show()
        fig.savefig(attr)