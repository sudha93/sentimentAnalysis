
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim 
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np 
import sys
import csv 
import pickle
import time 
# This code performs sentiment analysis on movie reviews (Rotten Tomatoes Dataset)
# using logistic regression 
# Preprocessing  

traindata = "/home/star/passion/Marathon/senti_analysis/train.tsv"
#testdata = "/home/star/passion/Marathon/senti_analysis/test.tsv"
glove = "/home/star/passion/Marathon/senti_analysis/glove.6B/glove.6B.50d.txt"

def preprocess(filepath) :
    # Preprocess and store each element in a list 
    input_data = []
    labels = [] 
    
    with open(filepath) as f:
        lines = f.readlines()
        del lines[0]
        #print lines[34:40]
        for line in lines:
            #print line 
            line = line.split('\t')
            #n = len(line[2].split())
            #if (n>=2) and (type(line[3]) is int):
            if len(line) >=4:
                input_data.append(line[-2])
               # print line[2]
                #print line[3]
                labels.append(int(line[-1].strip())) 

    return input_data, labels 
    '''
    rowlist = []
    with open(filepath,'rb') as f:
        data = csv.reader(f)
        for row in data:
            rowlist.append(row)
        print rowlist[34:40]
    '''
        


# 
def convert(lst,glove):
    final = []
    toDelete = []
    for i,item in enumerate(lst):
        #print item
        l = []
        #vector_sum = np.zeros(50)
        s = item.split()
        for token in s:
            if token.lower() in glove.vocab.keys():
                #print token
                #n+= 1
                vector = glove[token.lower()]
                l.append(vector)
                # element wise addition
        l = np.array([np.array(x) for x in l])
        vector_sum = l.sum(0)
        #print l 
        #print n
        if len(l)== 0 :
            toDelete.append(i)
        else :
            avg = np.divide(vector_sum,float(len(l)))
            #print avg
            final.append(avg) 
    return final,toDelete 

def delElements(lst,delList):
    if len(delList) > 0 :        
        for index in sorted(delList, reverse=True) :
            del lst[index]

'''
#MAIN CODE 
start_time = time.time()
phrases,labels = preprocess(traindata)
# Stop words list 

N = 10000
# Picking a part of the total data for test set 
train,test,trainLabels, testLabels = train_test_split(phrases[:N], labels[:N], test_size=0.25,random_state=48 )

#print phrases[:10]
#print labels[:10]

 
# Convert the glove vectors into word2vec file format with parameters 'dim of wordvectors'
# and 'no.of words' added 
glove2word2vec(glove_input_file= glove, word2vec_output_file="gensim_glove_vectors.txt")

glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
#print glove_model.vocab['blahdbbddb']
#print type(glove_model.vocab)
#print glove_model.vocab.get('boy')

print("--- %s seconds ---" % (time.time() - start_time))
m = time.time()
train, d1 = convert(train,glove_model)
test , d2 = convert(test,glove_model)
print("--- %s seconds ---" % (time.time() - m))
l = time.time()

# storing values
pickle.dump(train,open('train.pkl','w'))
pickle.dump(test,open('test.pkl','w'))
pickle.dump(trainLabels,open('trainLabels.pkl','w'))
pickle.dump(testLabels,open('testLabels.pkl','w'))
pickle.dump(d1,open('d1.pkl','w'))
pickle.dump(d2,open('d2.pkl','w'))

print 'completed'
sys.exit(0)
'''

train = pickle.load(open('train.pkl'))
test = pickle.load(open('test.pkl'))
trainLabels = pickle.load(open('trainLabels.pkl'))
testLabels = pickle.load(open('testLabels.pkl'))
d1 = pickle.load(open('d1.pkl'))
d2 = pickle.load(open('d2.pkl'))

delElements(trainLabels,d1)
delElements(testLabels,d2)

#test, goldLabels = preprocess(testdata,1)
 

# Logistic Regression 
''' Here there are five classes , from 0 to 4 '''

model = LogisticRegression()
model.fit(train,trainLabels)
op = model.predict(test)
acc_score = accuracy_score(testLabels,op)*100 
print 'accuracy:',acc_score

#print("--- %s seconds ---" % (time.time() - l))





                






















