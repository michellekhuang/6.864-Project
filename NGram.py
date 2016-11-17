import nltk
import re
import string

# http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
def split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
    
# http://stackoverflow.com/questions/743806/split-string-into-a-list-in-python/17951315#17951315
def get_words(text):
    return [word.strip(string.punctuation) for word in text.split()]
    
# Extracts and necesary data for the bigram model including
#       1. frequency of words in the corpus
#       2. frequency of transitions between words, START, and END tags
#   and then uses the countes to calculate probability of a transition
#   using maximum likelyhood
#
# Returns: dict named frequency where frequency[a] = # occurences of a in the corpus 
#          dict of dict named transition, where transition[a][b] = p(b follows a | a)        

def get_bigram_data(training_data):
    frequency = {'END_OF_SENTENCE': 1}
    transition = {}
    previous_word = 'END_OF_SENTENCE'
    
    # training data should be in readable format (ex. TAR)
    with open(training_data) as f:
        
        junk = False
        text = f.read()
        sentences = split_into_sentences(text)
        
        for sentence in sentences:           
            # remove legal text junk in the intro
            if '*END*' in sentence:
                junk = False
                continue
            
            if junk:
                continue
            
            if '\x00' in sentence:
                junk = True
                continue
            
            # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
            words = get_words(sentence)
            
            for word in words:
                # update frequency dictionary
                if word not in frequency:
                    frequency[word] = 0
                frequency[word] += 1
                
                # update transition dictionary
                if word not in transition:
                    transition[word] = {}
                if previous_word not in transition[word]:
                    transition[word][previous_word] = 0
                transition[word][previous_word] += 1
                
                previous_word = word
            previous_word = 'END_OF_SENTENCE'
                            
    for word in transition:
        for next in transition[word]:
            transition[word][next] = float(transition[word][next])/frequency[word]
            assert transition[word][next] >= 0 and transition[word][next] <= 1
    
    return frequency, transition
    
# Extracts from test data the answers questions into an answer dict where
#   answer['10'] gives the multiple choice answer to question 10
# Extracts from test data the problem statement into a problem dict where
#   problem['10']['statement'] gives the problem statement for question 10
#   problem['10'][letter] gives the answer corresponding to the letter such that
#   letter is an element in ['a', 'b', 'c', 'd', 'e']
# Returns: answer dict and problem dict 
    
def get_test_data(test_data):
    answer = {}
    question = {}
    
    with open(test_data) as f:
    
        # skip first 20 junk lines
        for _ in range(19):
            next(f)
        
        # extract the correct answers
        for line in f:
                
            # first line messed up     
            if '\x00' in line:
                line = '1) [d] swear'
                
            data = line.split()
            q_num = data[0].strip(')')
            q_ans = data[1][1]
            answer[q_num] = q_ans
            
            # there are exactly 1040 questions in the test set
            if q_num == '1040':
                break
        
        expecting = ('QUESTION', 1)
        q_num = 0
        
        # extract the question data
        for line in f:
            
            # first question line is messed up
            if '\x00' in line:
                line = '1) I have seen it on him , and could _____ to it.'
            
            words = line.split()
            
            # skip empty lines
            if len(words) == 0:
                continue
            
            # Format of problems should be a question statement followed by
            # 5 answer choices for to fill in the blank for the question statement
            
            if expecting[0] == 'QUESTION':
                q_num = words[0].strip(')')
                sentence = ' '.join(words[1:])
                question[q_num] = {}
                question[q_num]['statement'] = sentence
                expecting = ('ANSWER', 5)
                
            elif expecting[0] == 'ANSWER':
                ans_choice = words[0].strip(')')
                question[q_num][ans_choice] = words[1]
                expecting = ('ANSWER', expecting[1] - 1)
                
                # all five answer options were seen, look for question statement next
                if expecting[1] == 0:
                    expecting = ('QUESTION', 1)
                    
                    # should stop reading after completing all 1040 test questions
                    if q_num == '1040':
                        break
                        
    return question, answer
    
frequency, transition_prob = get_bigram_data('dataset/holmes_Training_Data.tar')
question, answer = get_test_data('dataset/MSR_Sentence_Completion_Challenge_V1.tar')
