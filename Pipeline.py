import itertools
import pdb

class Pipeline:
    def __init__(self, steps):
        self.steps = steps[:-1]
        self.model = steps[-1]
        # TODO implement some checks to make sure steps are proper

    def _iter(self):
        stop = len(self.steps)
        for idx, step in enumerate(itertools.islice(self.steps, 0, stop)):
            yield idx, step 

    def transform(self, X, **fit_params):
        # make a copy
        Xt = X
        # iterate through the pipeline
        for (idx, step) in self._iter():
            #pdb.set_trace()
            Xt = step.transform(Xt)
        # scikit pipeline uses idx to update self.steps because it uses 
        # a clone of the transformer, but in our case we don't need to yet
        return Xt

    def fit_transform(self, X, y, **fit_params):
        # make a copy
        Xt = X
        # iterate through the pipeline
        for (idx, step) in self._iter():
            Xt = step.fitTransform(Xt)
        # scikit pipeline uses idx to update self.steps because it uses 
        # a clone of the transformer, but in our case we don't need to yet

        # fit our model
        self.model.fit(Xt, y)
        return Xt
    
    def predict(self, X, y, *args):
        # make a copy
        Xt = self.transform(X)
        
        # make ur prediction
        return self.model.predict(Xt, y, *args)

    def validate(self, predictions, labels, *args):
        self.model.validate(predictions, labels, *args)