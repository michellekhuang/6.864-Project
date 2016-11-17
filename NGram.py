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
#  1. frequency of words in the corpus
#  2. frequency of transitions between words, START, and END tags
# and then uses the countes to calculate probability of a transition
# using maximum likelyhood
#
# returns: dict of dict named transition, where transition[a][b] = p(b follows a | a)

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
    
    return transition
    
transition_prob = get_bigram_data('dataset/holmes_Training_Data.tar')