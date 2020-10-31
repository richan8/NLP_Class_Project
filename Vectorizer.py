'''
Modes:
    TFIDF
    Count
'''
class Vectorizer:
    vec = None
    mode = None

    def __init__(self, trainData, mode = 'TFIDF'):
        if(mode == 'TFIDF'):
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer()
        elif(mode == 'Count'):
            from sklearn.feature_extraction.text import CountVectorizer
            self.vec = CountVectorizer()
        self.mode = mode

    def __str__(self):
        return('Custom Vectorizer using '+self.mode+' Vectorization')

    def features(self):
        return(self.vec.get_feature_names())

    def fitTransform(self, data):
        return(self.vec.fit_transform(data))

    def transform(self, data):
        return(self.vec.transform(data))