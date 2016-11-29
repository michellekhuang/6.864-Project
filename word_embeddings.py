from metrics import *
from model import Model
from NGram import get_test_data

# IMPORTANT: Include vectors.txt file with word embeddings from PSET4 at the same level
#            as the code here.

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

# Using different metrics, this function chooses the word which is closest to 
# the sentence statmeent with a blank
# 
# Inputs: q_info = question[q_num] that contains the statement and word options
#         m - contains information about word embeddings
#         metric - determines what metric to use to determine closeness
# Output: best answer choice for the input question
#
# Test Results (on 1040 test samples):
#   30.7% accuracy using wmd
#   30.3% accuracy using rwmd
#   31.25% accuracy using wcd
#   18.8% accuracy using cos
#   19.0% accuracy using euc

def find_best_answer(q_info, m, metric):
    s = q_info['statement']
    distances = []
    answers = 'abcde'
    for answer in answers:
        try:
            distances.append(metric(s, q_info[answer], m))
        except:
            return 'a'
    best_index = distances.index(min(distances))
    return answers[best_index]
    
# Tries to determine closeness of options to specific non-stopwords in the sentence
# Considering changing to phrases
# Really bad, ~17% accuracy for 100 samples
def find_best_answer_2(q_info, m, metric):
    s = q_info['statement']
    words = s.split()
    words = [x for x in words if x not in STOPWORDS]
    distances = []
    answers = 'abcde'
    for answer in answers:
        d = 0.0
        for word in words:
            try:
                d += metric(word, q_info[answer], m)
            except:
                continue
        distances.append(d)
    best_index = distances.index(min(distances))
    return answers[best_index]

print 'loading test data...'

question, answer = get_test_data('dataset/MSR_Sentence_Completion_Challenge_V1.tar')

print 'loading model data...'

m = Model('./vectors.txt')
m.load()

right = 0.0
wrong = 0.0

print 'predicting answers... '

for i in range(1, len(question)+1):
    q_num = str(i)
    best = find_best_answer(question[q_num], m, euc)
    if best == answer[q_num]:
        right += 1
    else:
        wrong += 1
        
print 'The accuracy is ' + str(right/(right + wrong))
