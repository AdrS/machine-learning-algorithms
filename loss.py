from statistics import average, median, sign

class LossFn:

    def prior(self, Y):
        '''Returns the optimal baseline prediction for a training set.'''
        raise NotImplementedError

    def __call__(self, targets, predictions):
        return self.loss(targets, predictions)

    def loss(self, targets, predictions):
        raise NotImplementedError

    def negative_gradients(self, targets, predictions):
        '''Returns list of gradients of the loss wrt each prediction'''
        raise NotImplementedError

def residuals(targets, predictions):
    return [t - p for t, p in zip(targets, predictions)]

class RegressionLossFn(LossFn):

    def residual_loss(self, residuals):
        '''Computes the loss from the residual = target - predicted'''
        raise NotImplementedError

    def residual_gradients(self, residuals):
        '''Computes gradients of loss wrt the residuals target - predicted'''
        raise NotImplementedError

    def loss(self, targets, predictions):
        return self.residual_loss(residuals(targets, predictions))

    def negative_gradients(self, targets, predictions):
        '''Computes negative gradients of loss wrt predictions'''
        return self.residual_gradients(residuals(targets, predictions))

class MeanSquaredError(RegressionLossFn):

    def prior(self, Y):
        return average(Y)

    def residual_loss(self, residuals):
        return average([r*r for r in residuals])

    def residual_gradients(self, residuals):
        return [2*r for r in residuals]


class MeanAbsoluteError(RegressionLossFn):

    def prior(self, Y):
        return median(Y)

    def residual_loss(self, residuals):
        return average([abs(r) for r in residuals])

    def residual_gradients(self, residuals):
        return [sign(r) for r in residuals]

class HuberLossFn(RegressionLossFn):

    def __init__(self, delta=1):
        self.delta = delta

    def prior(self, Y):
        # This approximation assumes |y| > delta for most of Y
        return median(Y)

    def residual_loss(self, residuals):
        def L(r):
            if abs(r) <= self.delta:
                return 0.5*r*r
            return self.delta*(abs(r) - 0.5*self.delta)
        return average([L(r) for r in residuals])

    def residual_gradients(self, residuals):
        def G(r):
            if abs(r) < self.delta:
                return r
            return self.delta*sign(r)
        return [G(r) for r in residuals]
