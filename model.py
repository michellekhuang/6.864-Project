import numpy as np


class Model(object):
    def __init__(self, path):
        self.path = path
        self.vocab = {}
        self.matrix = None

    def load(self):
        '''
        Loads the word vectors into memory.
        '''
        
        vectors = {}
        with open(self.path, 'r') as f:
            for i, line in enumerate(f):
                word, vector = line.strip().split('\t')
                self.vocab[word] = i
                vectors[word] = vector
        self.matrix = np.empty((len(self.vocab), 300))
        for word in self.vocab:
            self.matrix[self.vocab[word]] = np.fromstring(vectors[word][1:-1], sep=' ')
        del vectors

    def __getitem__(self, key):
        '''
        Return a word vector or subset of word vectors depending on the key.
        If `key` is a word string, returns the word vector for that word if present,
        else throws and error. If `key` is a list of word strings, returns a matrix X
        where
        
            X[i] = word_vector(key[i])

        else throws an error if any of the word strings does not have a vector.

        Example:
        m = Model('vectors.txt'); m.load()
        m['hello'] # returns a single vector of shape (300,)
        m[['hello', 'world']] # returns a matrix with shape (2, 300)
        '''
        
        if isinstance(key, str):
            return self.matrix[self.vocab[key]]
        elif isinstance(key, list):
            indices = map(lambda w: self.vocab[w], key)
            return self.matrix[indices]
        else:
            raise Exception('Invalid key: {}'.format(key))

    def __contains__(self, word):
        return word in self.vocab
