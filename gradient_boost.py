import random

from statistics import average, variance

def sample_batch(X, Y, fraction):
    'Returns a subset with floor(fraction*|X|) of the training examples.'
    sample_idxs = random.sample(range(len(X)), k=int(len(X)*fraction))
    return [X[i] for i in sample_idxs], [Y[i] for i in sample_idxs]

class GradientBoostModel:
    def __init__(self, create_base_model, loss_fn,
            num_models=50,
            learning_rate=0.1,
            sample_fraction=None):
        # TODO: step length
        self.create_base_model = create_base_model
        self.loss_fn = loss_fn
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.models = []
        self.sample_fraction = sample_fraction

    def fit(self, X, Y):

        class Constant:
            def __init__(self, value):
                self.value = value 
            def predict(self, X):
                return [self.value]*len(X)
        self.models.append(Constant(self.loss_fn.prior(Y)))

        for i in range(self.num_models):
            if self.sample_fraction:
                X_batch, Y_batch = sample_batch(X, Y, self.sample_fraction)
            else:
                X_batch, Y_batch = X, Y
            X_pred = self.predict(X_batch)
            residuals = self.loss_fn.negative_gradients(
                Y_batch, X_pred)
            model = self.create_base_model(loss_fn=self.loss_fn)
            model.fit(X_batch, residuals)
            self.models.append(model)

    def aggregate(self, predictions):
        raise NotImplementedError

    def predict(self, X):
        # List of model predictions for each element of X
        predictions = zip(*[model.predict(X) for model in self.models])
        # Aggregate the predictions from all the models into a single
        # prediction
        return [self.aggregate(item_preds) for item_preds in predictions]

class GradientBoostRegressor(GradientBoostModel):

    def aggregate(self, predictions):
        return predictions[0] + self.learning_rate*sum(predictions[1:])
