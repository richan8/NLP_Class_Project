from Vectorizer import Vectorizer
from Model import Model
from Downloader import Downloader
from Pipeline import Pipeline
from Tokenizer import Tokenizer
from Stemmer import Stemmer
from Splitter import splitRows
import pdb
import numpy as np
import pandas as pd
import sys, getopt

'''
Arguments (comma separated, in order):
    Data Transformations:
        toke                -> Tokenizer
        stem                -> Stemmer [Optional]
        split-sentences     -> Split rows via sentences
        vect                -> Vectorizer

    Model Selection:
        svm                 -> Sets model as SVM 
        nb                  -> Sets model as NB

    Misc:
        no-verb             -> (No Verbose): Command to not print intermediate steps to terminal.

Example:
    python main.py toke,stem,vect,nb
'''

### Globals
verbose = True

def run(vectorizer, model, train_data, train_labels, test_data, test_labels):
    vecTrainData = v.fitTransform(trainData)
    vecTestData = v.transform(testData)
    clf.fit(vecTrainData, trainLabels)
    clf.predict(vecTestData, testLabels, verbose = True)
    return 0

def main(argv):
    # construct our pipeline list reading from command line args
    # still need to figure out best way to pass parameters on command
    # line

    global verbose
    split = None

    transforms = []
    for arg in argv[0].split(","):
        if arg == "toke":
            transforms.append(Tokenizer())
        elif arg == "stem":
            transforms.append(Stemmer())
        elif arg == "vect":
            transforms.append(Vectorizer(None))
        elif arg == "svm":
            transforms.append(Model('svm'))
        elif arg == "nb":
            transforms.append(Model('nb'))
        elif arg == "no-verb":
            verbose =  False
        elif arg == "split-sentences":
            split = "sentences"

    pipe = Pipeline(transforms)

    # read our data (hardcoded for now)
    df0 = pd.read_pickle("./data/democrat_comments.pkl")
    df1 = pd.read_pickle("./data/republican_comments.pkl")
    
    if(split is not None):
        if(verbose):
            print('Splitting Democrat comments')
        df0 = splitRows(df0, mode=split, verbose=verbose)

        if(verbose):
            print('Splitting Republican comments')
        df1 = splitRows(df1, mode=split, verbose=verbose)

    label0 = df0.subreddit.iloc[0]
    label1 = df1.subreddit.iloc[0]

    # concatenate and clean our data
    X = pd.concat([df0.body, df1.body], ignore_index=True)
    y = pd.concat([df0.subreddit, df1.subreddit], ignore_index=True).replace(to_replace=[label0, label1], value=[0, 1])

    # split into training and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if(verbose):
        print('Applying Transforms and Training Model')
    # fit our data
    pipe.fit_transform(X_train, y_train)

    # do the prediction
    pipe.predict(X_test, y_test)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("done")