import nltk
import re
import string
import os
    
# Extracts and necesary data for the bigram model including
#       1. frequency of words in the corpus
#       2. frequency of transitions between words, START, and END tags
#   and then uses the countes to calculate probability of a transition
#   using maximum likelyhood
#
# Returns: dict named frequency where frequency[a] = # occurences of a in the corpus 
#          dict of dict named transition, where transition[a][b] = p(b follows a | a)        

def get_bigram_data(training_data_folder):

    frequency = {'END_OF_SENTENCE': 1}
    transition = {}
    previous_word = 'END_OF_SENTENCE'
    
    # go through all text files in training data folder (TBTAS10.TXT, MOHIC10.TXT, ACHOE10.TXT)
    for root, dirs, files in os.walk(training_data_folder):
        for i, file in enumerate(files):
            print os.path.join(root, file)
            with open(os.path.join(root, file)) as f:
                
                print 'Starting ' + file + ' which is file ' + str(i) + ' out of ' + str(len(files))
                
                text = f.read()
                
                # TODO: Figure out why some texts are throwing errors for this
                try:
                    sentences = [nltk.tokenize.word_tokenize(s) for s in nltk.tokenize.sent_tokenize(text)]
                except:
                    print 'ERROR READING FILE'
                    continue
                    
                for sentence in sentences:
                    for word in sentence:
                        word = word.lower()
                        
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
    
def get_test_data(test_data_folder):

    answer = {}
    question = {}

    # extract the correct answers
    with open(os.path.join(test_data_folder, 'Holmes.human_format.answers.txt')) as f:
        for line in f:
                
            data = line.split()
            q_num = data[0].strip(')')
            q_ans = data[1][1]
            answer[q_num] = q_ans
            
    with open(os.path.join(test_data_folder, 'Holmes.human_format.questions.txt')) as f:
        expecting = ('QUESTION', 1)
        q_num = 0
        
        # extract the question data
        for line in f:
        
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
                        
    return question, answer
    
# The bigram chooses the best option based on highest transition probability
#   from the previous word. If no transition has been seen before, it picks the word
#   with the greatest frequency of occurence in the corpus
# Returns: percent correct, model answers

def get_bigram_results(frequency, transition_prob, question, answer):

    options = 'abcde'
    model_answer = {}
   
    for q_num in question:
    
        # extract the word before the blank
        statement = question[q_num]['statement']
        words = statement.split()
        blank_index = words.index('_____')
        previous_word = 'END_OF_SENTENCE'
        if blank_index > 0:
            previous_word = words[blank_index-1].lower()
            
        # find the best option based on highest transition probability
        best_option = ''
        p_best_option = 0
        for option in options:
            option_word = question[q_num][option]
            if previous_word in transition_prob:
                if option_word in transition_prob[previous_word]:
                    p_option_word = transition_prob[previous_word][option_word]
                    if p_option_word > p_best_option:
                        p_best_option = p_option_word
                        best_option = option
        
        # if no transition seen before, pick the option which had occured most often
        highest_freq = 0
        if best_option == '':
            for option in options:
                option_word = question[q_num][option]
                if option_word in frequency:
                    freq_option_word = frequency[option_word]
                    if freq_option_word > highest_freq:
                        highest_freq = freq_option_word
                        best_option = option
        
        model_answer[q_num] = best_option
                
    # calculate accuracy of model
    correct = 0
    total = 0
    for q_num in answer:
        total += 1
        if model_answer[q_num] == answer[q_num]:
            correct += 1
    
    percent_correct = float(correct)/total

    return percent_correct, model_answer
    
def run_bigram_model():
    print 'Extracting Training Data...'
    frequency, transition_prob = get_bigram_data('dataset/Holmes_Training_Data/')
    print 'Extracting Test Data...'
    question, answer = get_test_data('dataset/SAT_Questions')
    print 'Computing Model Results...'
    percent_correct, model_answers = get_bigram_results(frequency, transition_prob, question, answer)
    print 'Percent Correct: ' + str(percent_correct)

if __name__ == '__main__':
    run_bigram_model()

