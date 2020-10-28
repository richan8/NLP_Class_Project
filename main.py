from Vectorizer import Vectorizer
from Model import Model
import numpy as np
import pandas as pd

### Global Vars
dataPath = 'data/sample.csv'
trainSplitRatio = 0.8

### Loading the data
'''
trainData = np.array(['The','brown', 'The fox', 'The quick'])
trainLabels = np.array([1,0,0,1])
testData = np.array(['The  quick', 'The brown','brown fox'])
testLabels = np.array([1,0,0])
'''
data = pd.read_csv(dataPath)[['tweet','label']].to_numpy()
np.random.shuffle(data)
n = int(trainSplitRatio * len(data))
trainData = data[:n,0]
trainLabels = data[:n,1]
testData = data[n:,0]
testLabels = data[n:,1]

print('Train data:\t',trainData.shape)
#print(trainData[:3])
print('Train Labels:\t',trainLabels.shape)
#print(trainLabels[:3])
print('Test data:\t',testData.shape)
#print(testData[:3])
print('Test Labels:\t',testLabels.shape)
#print(testLabels[:3])

### Initializing and using our vectorizer
v = Vectorizer(trainData, mode='TFIDF')
vecTrainData = v.fitTransform(trainData)
vecTestData = v.transform(testData)

'''
print(trainData[:3])
print(testData[:3])
print('\n--------------')
print(len(v.features()))
print(vecTrainData.toarray())
print('')
print(vecTestData.toarray())
'''

### Initializing, Training and Validating the model
clf = Model()
clf.fit(vecTrainData, trainLabels)
clf.predict(vecTestData, testLabels, verbose = True)