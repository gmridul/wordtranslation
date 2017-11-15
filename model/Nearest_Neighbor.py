
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.neighbors import NearestNeighbors


# In[7]:


class Nearest_Neighbor(object):
    word_embedding_dict = dict()
    word_embedding_nn = dict()
    def __init__(self, embeddings_dir, lang_list, k_neighbors):
        for lang in lang_list:
            file_path = embeddings_dir+'/wiki.'+lang + '.vec'
            #cur_word_embeddings = self.load_word_embeddings(file_path)
            #print (np.shape(cur_word_embeddings))
            Nearest_Neighbor.word_embedding_dict[lang] = self.load_word_embeddings(file_path)
            Nearest_Neighbor.word_embedding_nn[lang] = NearestNeighbors(                            n_neighbors=k_neighbors, algorithm='ball_tree', metric='euclidean').fit(Nearest_Neighbor.word_embedding_dict[lang])

    def load_word_embeddings(self, file_path):
        res = []
        with open(file_path, encoding='utf-8') as f:
            next(f)
            for line in f:
                vec = line.split(' ')
                res.append([float(x) for x in vec[1:-1]])
        return np.array(res)
        
    # word_vec is 2D vector list of words.
    def knn(self, word_vec, lang, k=10):
        #assert input word_vec is 2D 
        #assert language is one of the keys
        #assert k < 10 
        dist, index = Nearest_Neighbor.word_embedding_nn[lang].kneighbors(word_vec)
        return Nearest_Neighbor.word_embedding_dict[lang][index[:, 0:k]]
        
        


# In[8]:
# Example Usage

"""
n = Nearest_Neighbor ("D:/UCSD/F17/CSE293/", ["pt"], 10)
point = Nearest_Neighbor.word_embedding_dict['pt'][0:10]
res = n.knn(point, "pt", 5)


#Nearest_Neighbor.word_embedding_dict['pt']
#knn(word_embedding_dict['pt'][ind[:, 0:5]]

"""