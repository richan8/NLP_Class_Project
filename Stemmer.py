'''
Modes:
    Porter
    Lancaster
    Lemmatize

    https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_stemming_lemmatization.htm
'''
class Stemmer:
    stemmer = None
    mode = None

    def __init__(self, mode = 'Porter'):
        if(mode == 'Porter'):
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        elif(mode == 'Lancaster'):
            from nltk.stem import LancasterStemmer
            self.stemmer = LancasterStemmer()
        elif(mode == 'Lemmatize'):
            from nltk.stem import WordNetLemmatizer
            self.stemmer = WordNetLemmatizer()
        elif(mode == 'Snowball'):
            raise Exception("TODO")
        elif(mode == 'Regexp'):
            raise Exception("TODO")
        self.mode = mode

    def __str__(self):
        return('Custom Vectorizer using '+self.mode+' Vectorization')

    # data is a list of words
    def fitTransform(self, data):
        return self.transform(data)

    # data is a list of words
    def transform(self, data):
        if(self.mode == "Lemmatize"):
            return([self.stemmer.lemmatize(word) for word in data])
        else:
            return([self.stemmer.stem(word) for word in data])