
# coding: utf-8

# In[29]:


import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as sk
import glob
import os


# In[2]:


class Nearest_Neighbor(object):
    word_embedding_dict = dict()
    word_embedding_nn = dict()
    def __init__(self, embeddings_dir, domain_dir, lang_list, k_neighbors, flag='train', normalized=False):
        for lang  in lang_list:
            file_path = embeddings_dir+'/wiki.'+lang + '.vec'
            #cur_word_embeddings = self.load_word_embeddings(file_path)
            #print (np.shape(cur_word_embeddings))
            Nearest_Neighbor.word_embedding_dict[lang] = self.load_word_embeddings(file_path, domain_dir, lang, flag, normalized)
            Nearest_Neighbor.word_embedding_nn[lang] = NearestNeighbors(                            n_neighbors=k_neighbors, algorithm='ball_tree', metric='euclidean').fit(Nearest_Neighbor.word_embedding_dict[lang])

    def load_word_embeddings(self, file_path, domain_dir, lang, flag, normalized):
        res = []
        words_in_domain = {}
        glob_dir = ""
        
        if(flag == 'test'):
            glob_dir = domain_dir+ '*' + lang + '*.txt'
        else: #train
            glob_dir = domain_dir+ '*' + lang + '*train.txt'
        
        for file in glob.glob(glob_dir):
            file_name = (os.path.basename(file))
            lang_pair = file_name.split('_')[1].split('.')[0].split('-')
            index= -1
            if lang == lang_pair[0]: 
                index = 0
            elif lang == lang_pair[1]:
                index = 1
            with open(file, 'r') as f:
                for line in f:
                    word = line.split(' ')[index].strip()
                    words_in_domain[word] = True
        
        with open(file_path, encoding='utf-8') as f:
            next(f)
            for line in f:
                vec = line.split(' ')
                if vec[0].strip() in words_in_domain:
                    #print(vec[0].strip())
                    res.append([float(x) for x in vec[1:-1]])
        if normalized:
            return np.array(sk.normalize(res))
        else:
            return np.array(res)
    # word_vec is 2D vector list of words.
    def knn(self, word_vec, lang, k=10):
        #assert input word_vec is 2D 
        #assert language is one of the keys
        #assert k < 10 
        dist, index = Nearest_Neighbor.word_embedding_nn[lang].kneighbors(word_vec)
        return Nearest_Neighbor.word_embedding_dict[lang][index[:, 0:k]]
        
        


# ### Usage
# n = Nearest_Neighbor (embeddings_dir="D:/UCSD/F17/CSE293/", domain_dir='D:/UCSD/F17/CSE293/wordtranslation/model/data/', lang_list=['pt', 'en'], k_neighbors=10)
# point = Nearest_Neighbor.word_embedding_dict['pt'][0:10]
# np.shape(n.knn(point, "pt", 5))

# In[64]:


#nearest_ne = Nearest_Neighbor(embeddings_dir="D:/UCSD/F17/CSE293/", domain_dir='D:/UCSD/F17/CSE293/wordtranslation/model/data/', lang_list=['dummy'], k_neighbors=3)


# In[66]:


#nearest_ne.knn([[1, 2, 3]], "dummy", 2)
#Nearest_Neighbor.word_embedding_dict

