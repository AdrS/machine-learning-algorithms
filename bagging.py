import random

from collections import Counter

def boostrap_sample(X, Y, k):
    sample_idxs = random.choices(range(len(X)), k=k)
    return [X[i] for i in sample_idxs], [Y[i] for i in sample_idxs]

class BaggingModel:
    def __init__(self, create_base_model, num_models=50):
        self.create_base_model = create_base_model
        self.num_models = num_models
        self.models = []

    def fit(self, X, Y):
        for i in range(self.num_models):
            X_bootstrap, Y_bootstrap = boostrap_sample(X, Y, k=len(X))
            model = self.create_base_model()
            model.fit(X_bootstrap, Y_bootstrap)
            self.models.append(model)

    def aggregate(self, predictions):
        raise NotImplementedError

    def predict(self, X):
        # List of model predictions for each element of X
        predictions = zip(*[model.predict(X) for model in self.models])
        # Aggregate the predictions from all the models into a single
        # prediction
        return [self.aggregate(item_preds) for item_preds in predictions]

class BaggingClassifier(BaggingModel):

    def aggregate(self, predictions):
        # Each classifier votes and the prediction with the most votes wins.
        votes = Counter(predictions)
        return votes.most_common(1)[0][0]

class BaggingRegressor(BaggingModel):

    def aggregate(self, predictions):
        return sum(predictions)/len(predictions)
