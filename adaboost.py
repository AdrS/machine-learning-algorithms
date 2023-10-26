import math
from statistics import average, median

def transform_to_minus1_positive1(Y):
    Z = []
    for y in Y:
        if y == 0 or y == -1:
            Z.append(-1)
        elif y == 1:
            Z.append(1)
        else:
            raise ValueError('Inputs must have values {0, 1} or {-1, 1}')
    return Z

class AdaBoostModel:
    def __init__(self, create_base_model, num_models=50):
        self.create_base_model = create_base_model
        self.num_models = num_models
        self.models = []
        self.alphas = []

    def transform_input(self, X, Y):
        return X, Y

    def update_weights(self, targets, predictions, weights):
        'Returns the weight of the model and the new weights of the data'
        raise NotImplementedError

    def fit(self, X, Y):
        X, Y = self.transform_input(X, Y)
        weights = [1/len(X)]*len(X)
        for i in range(self.num_models):
            model = self.create_base_model()
            model.fit(X, Y, weights)
            predictions = model.predict(X)
            alpha, weights = self.update_weights(Y, predictions, weights)
            self.models.append(model)
            self.alphas.append(alpha)

    def aggregate(self, item_preds):
        raise NotImplementedError

    def predict(self, X):
        # List of model predictions for each element of X
        predictions = zip(*[model.predict(X) for model in self.models])
        return [self.aggregate(item_preds) for item_preds in predictions]

class AdaBoostClassifier(AdaBoostModel):

    def transform_input(self, X, Y):
        return X, transform_to_minus1_positive1(Y)

    def update_weights(self, targets, predictions, weights):
        epsilon = sum(int(p != t) for p, t in
            zip(predictions, targets))/len(targets)
        alpha = math.log((1 - epsilon)/epsilon)/2
        unormalized_weights = [
            w*math.exp(-alpha*y*prediction) for
            w, prediction, y in zip(weights, predictions, targets)
        ]
        normalization_factor = sum(unormalized_weights)
        weights = [w/normalization_factor for w in unormalized_weights]
        return alpha, weights

    def aggregate(self, item_preds):
        # Compute a weighted majority
        s = sum(a*p for a, p in zip(self.alphas, item_preds))
        if s < 0:
            return 0
        return 1
 
class AdaBoostRegressor(AdaBoostModel):
    # "Improving Regressors Using Boosting Techniques", Drucker 1997

    def update_weights(self, targets, predictions, weights):
        losses = [abs(t - p) for t, p in zip(targets, predictions)]
        D = max(losses)
        losses = [L/D for L in losses]
        #losses = [l**2/D for l in losses]
        L_avg = average(losses)
        beta = L_avg/(1 - L_avg)
        weights = [w*beta**(1 - L) for w, L in zip(weights, losses)]
        alpha = -math.log(beta)
        return alpha, weights

    def aggregate(self, item_preds):
        return median(item_preds, self.alphas)

# TODO: multi-class AdaBoost
