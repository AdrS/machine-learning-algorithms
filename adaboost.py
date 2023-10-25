import math

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

class AdaBoostClassifier:
    def __init__(self, create_base_model, num_models=50):
        self.create_base_model = create_base_model
        self.num_models = num_models
        self.models = []
        self.alphas = []

    def fit(self, X, Y):
        Y = transform_to_minus1_positive1(Y)
        weights = [1/len(X)]*len(X)
        for i in range(self.num_models):
            model = self.create_base_model()
            model.fit(X, Y, weights)
            predictions = model.predict(X)
            epsilon = sum(int(p != y) for p, y in zip(predictions, Y))/len(Y)
            alpha = math.log((1 - epsilon)/epsilon)/2

            unormalized_weights = [
                w*math.exp(-alpha*y*prediction) for
                w, prediction, y in zip(weights, predictions, Y)
            ]
            normalization_factor = sum(unormalized_weights)
            weights = [w/normalization_factor for w in unormalized_weights]

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # List of model predictions for each element of X
        predictions = zip(*[model.predict(X) for model in self.models])

        def weighted_majority(item_preds):
            s = sum(a*p for a, p in zip(self.alphas, item_preds))
            if s < 0:
                return 0
            return 1

        return [weighted_majority(item_preds) for item_preds in predictions]
 
# TODO: AdaBoostRegressor
# TODO: multi-class AdaBoost
