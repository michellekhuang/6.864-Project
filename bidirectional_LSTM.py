import sys
from NGram import get_test_data
from operator import add

test_data_location = 'dataset/MSR_Sentence_Completion_Challenge_V1/Data'
forward_file = 'forward_out_MSR.txt'
backward_file = 'forward_out_MSR.txt'
output_file = 'bidirectional_out_MSR.txt'

_, test_answer = get_test_data(test_data_location)

probs = {}
question_nums = []

with open(forward_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        values = line.split()
        # all data lines will have length 6
        if len(values) == 6:
            question_num = str(int(values[0]) + 1)
            question_nums.append(question_num)
            probs[question_num] = [float(x) for x in values[1:]]
        
with open(backward_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        values = line.split()
        if len(values) == 6:
            question_num = str(int(values[0]) + 1)
            for i in range(len(probs[question_num])):
                probs[question_num][i] += float(values[i+1])
                            
with open(output_file, 'w') as f:
    correct = 0.0
    total = 0.0
    
    for num in question_nums:
        total += 1
        choices = 'abcde'
        prediction = choices[probs[num].index(max(probs[num]))]
        if prediction == test_answer[num]:
            correct += 1
            
        f.write(str(num) + ' ')

        for i in range(5):
            f.write(str(probs[num][i]) + ' ')

        f.write('\n')
        
    f.write('Accuracy: ' + str(correct/total))
    
    print('Accuracy:', correct/total)
    
