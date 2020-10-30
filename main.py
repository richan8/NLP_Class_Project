from Vectorizer import Vectorizer
from Model import Model
from Downloader import Downloader
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
#downloader = Downloader()
# #data = downloader.fetch_subreddit("democrats")
#data = downloader.load_pickled_sub("./data/democrats_comments.pkl")
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

### Initializing, Training and Validating the model
clf = Model('svm')

def run(vectorizer, model, train_data, train_labels, test_data, test_labels):
    vecTrainData = v.fitTransform(trainData)
    vecTestData = v.transform(testData)
    clf.fit(vecTrainData, trainLabels)
    clf.predict(vecTestData, testLabels, verbose = True)
    return 0

run(v, clf, trainData, trainLabels, testData, testLabels)
