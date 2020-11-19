import numpy as np
import pandas as pd
'''
Splitting modes:
    words
    sentences
'''
def splitRows(data, mode = 'sentences', verbose = True):
    docs = [] #body
    scores = []
    subreddits = []

    if(verbose):
        print(data.head())

    if(mode == 'sentences'):
        for _,row in data.iterrows():
            rowDocs = [x for x in row['body'].split('.') if x != '']
            docs.extend(rowDocs)
            scores.extend([row['score']]*len(rowDocs))
            subreddits.extend([row['subreddit']]*len(rowDocs))

    res = pd.DataFrame()
    res['body'] = docs
    res['score'] = scores
    res['subreddit'] = subreddits
    if(verbose):
        print(res.head())
        print('Splitting complete, df length changed from %i to %i\n\n'%(len(data['body']),len(docs)))
    return(res)