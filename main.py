from Vectorizer import Vectorizer
from Model import Model
from Downloader import Downloader
from Pipeline import Pipeline
from Tokenizer import Tokenizer
from Stemmer import Stemmer
import pdb
import numpy as np
import pandas as pd
import sys, getopt

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
clf = Model('nb')

def run(vectorizer, model, train_data, train_labels, test_data, test_labels):
    vecTrainData = v.fitTransform(trainData)
    vecTestData = v.transform(testData)
    clf.fit(vecTrainData, trainLabels)
    clf.predict(vecTestData, testLabels, verbose = True)
    return 0

#run(v, clf, trainData, trainLabels, testData, testLabels)

# .\main.py toke,stem,vect,svm,nb
def main(argv):
    # construct our pipeline list reading from command line args
    # still need to figure out best way to pass parameters on command
    # line
    transforms = []
    for transform in argv[0].split(","):
        if transform == "toke":
            transforms.append(Tokenizer())
        elif transform == "stem":
            transforms.append(Stemmer())
        elif transform == "vect":
            transforms.append(Vectorizer(None))
        elif transform == "svm":
            transforms.append(Model('svm'))
        elif transform == "nb":
            transforms.append(Model('nb'))
    pipe = Pipeline(transforms)

    # read our data (hardcoded for now)
    df_republican = pd.read_pickle("./data/republican_comments.pkl")
    df_democrat = pd.read_pickle("./data/democrat_comments.pkl")
    X = pd.concat([df_democrat.body, df_republican.body], ignore_index=True)
    y = pd.concat([df_democrat.subreddit, df_republican.subreddit], ignore_index=True)
    
    # split into training and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit our data
    pipe.fit_transform(X_train, y_train)

    # do the prediction
    pipe.predict(X_test, y_test)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("done")

