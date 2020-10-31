import pdb
'''
Models:
    Naive Bayes: nb
'''
class Model:
    model = None
    modelType = None

    def __init__(self, modelType = 'nb', opt = None):
        if(modelType == 'nb'):
            from sklearn.naive_bayes import MultinomialNB
            self.model = MultinomialNB()
            self.modelType = 'Naive Bayes'
        if(modelType == 'svm'):
            from sklearn import svm
            kernelType = 'rbf' #default of svms
            if opt:
                kernel = opt
            self.model = svm.SVC(kernel=kernelType)
            self.modelType = 'Support Vector Machine'
    
    def __str__(self):
        return('Model of type: '+self.modelType+' - '+str(type(self.model)))

    def fit(self, trainData, trainLabels):
        return(self.model.fit(trainData, list(trainLabels)))
    
    def validate(self,predictions,labels,verbose = True):
        from sklearn.metrics import precision_recall_fscore_support
        results = precision_recall_fscore_support(labels, predictions)
        acc = (list(predictions==labels).count(True))/(len(predictions))
        if(verbose):
            print('Accuracy of the model:\t %0.3f'%(acc))
            print('Precision wrt. class 0:\t %0.3f'%(results[0][0]))
            print('Recall wrt. class 0:\t %0.3f'%(results[1][0]))
            print('F1 Score wrt. class 0:\t %0.3f'%(results[2][0]))

        return(results,acc)


    def predict(self, testData, testLabels = None, verbose = True):
        predictions = self.model.predict(testData)
        if(testLabels is not None):
            return(self.validate(list(testLabels), predictions, verbose = verbose))
