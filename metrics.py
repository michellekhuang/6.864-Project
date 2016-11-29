from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cosine, euclidean, cdist, pdist, squareform

from pprint import pprint

# from nltk
STOPWORDS = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
                 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
                 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don',
                 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
                 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
                 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or',
                 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should',
                 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
                 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
                 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
                 'who', 'whom', 'why', 'will', 'with', 'yo', 'your', 'yours', 'yourself', 'yourselves'])

### HELPERS ###

def preprocess(doc, m, vectorize=False):
    '''
    Performs some preprocessing on the
    document string `doc`.
    Always returns a 2-tuple.
    If `vectorize` is True, this returns a tuple
    where the first element is a nBOW vector v
    where
    
        v[i] = count(words[i], doc) / len(doc)

    and the second element in the list of
    unique words `words` in `doc`. If False,
    the first element is just the token list
    representation of `doc` and the second
    element is a set rather than a list.
    '''
    
    doc = doc.strip().split()
    words = set(doc) & set(m.vocab) - STOPWORDS
    if vectorize:
        words = list(words)
        doc = nbow(doc, words)
    return doc, words

def joint_preprocess(doc1, doc2, m):
    '''
    Jointly preprocess the documents `doc1`
    and `doc2` separate but compatible nBOW
    vectors. Returns a 3-tuple where the first
    element is the nBOW vector for `doc1`, the second
    element is the nBOW vector for `doc2`, and third
    element is the list of words. Each word in the list
    corresponds to the corresponding index in either
    document vector.
    '''
    
    doc1, words1 = preprocess(doc1, m, vectorize=False)
    doc2, words2 = preprocess(doc2, m, vectorize=False)
    words = list(words1 | words2)
    d1 = nbow(doc1, words)
    d2 = nbow(doc2, words)
    return d1, d2, words

def nbow(doc, words):
    '''
    Computes the normalized bag-of-words (nBOW) representation.
    Given a document as a list of tokens
    and a list of words, returns a vector v
    where

        v[i] = count(words[i], doc) / len(doc)
    '''
    
    cntr = Counter(doc)
    vector = np.empty((len(words),))
    total = 0
    for i, word in enumerate(words):
        count = cntr[word]
        vector[i] = count
        total += count
    vector /= float(total)
    return vector

def compute_dist_matrix(m, words):
    '''
    Given a list of words, computes the distance between
    any pair returns the distances as a matrix X where
    
        X[i,j] = ||m[words[i]] - m[words[j]]||
    '''
    
    matrix = m[words]
    dist_matrix = squareform(pdist(matrix, metric='euclidean'))
    return dist_matrix

def compute_bipartite_dist_matrix(m, words1, words2):
    '''
    Given two lists of words, computes the distance between
    any pair across the two lists and returns the distances
    as a matrix X where
    
        X[i,j] = ||m[words1[i]] - m[words2[j]]||
    '''
    
    matrix1 = m[words1]
    matrix2 = m[words2]
    dist_matrix = cdist(matrix1, matrix2, metric='euclidean')
    return dist_matrix


### METRICS ###

def wmd(doc1, doc2, m):
    '''
    Word Mover's Distance (WMD)
    '''
    
    d1, d2, words = joint_preprocess(doc1, doc2, m)
    W = len(words)
    dist_matrix = compute_dist_matrix(m, words).reshape((W*W,))
    A = np.zeros((2*W, W*W))
    for i in xrange(W):
        A[i,i*W:(i+1)*W] = 1 # out flow constraint
        A[i+W,[i+(j*W) for j in xrange(W)]] = 1 # in flow constraint
    b = np.concatenate((d1, d2))
    result = linprog(dist_matrix, A_eq=A, b_eq=b)
    return result.fun

def rwmd(doc1, doc2, m):
    '''
    Relaxed Word Mover's Distance (RWMD)
    '''

    # will commit after pset is due
    pass
    

def wcd(doc1, doc2, m):
    '''
    Word Centroid Distance (WCD)
    '''

    # will commit after pset is due
    pass
    
def cos(doc1, doc2, m):
    '''
    Cosine Distance (COS)
    '''
    
    d1, d2, _ = joint_preprocess(doc1, doc2, m)
    return cosine(d1, d2)

def euc(doc1, doc2, m):
    '''
    Euclidean Distance (EUC)
    '''
    
    d1, d2, _ = joint_preprocess(doc1, doc2, m)
    return euclidean(d1, d2)
