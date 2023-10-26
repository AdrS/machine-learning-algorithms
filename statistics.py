import bisect

def average(X, weights=None):
    if not X:
        raise ValueError('Input must be non-empty')
    if weights:
        return sum(w*x for w, x in zip(weights, X))/sum(weights)
    else:
        return sum(X)/len(X)

def variance(X):
    mu = average(X)
    return sum(x*x for x in X)/len(X) - mu*mu

def cumulative_sum(X):
    'Returns an array A with A[i] = sum of the first i + 1 elements of X'
    previous = 0
    A = []
    for x in X:
        current = previous + x
        A.append(current)
        previous = current
    return A

def median(X, weights=None):
    if not X:
        raise ValueError('Input must be non-empty')
    if weights:
        X, weights = zip(*sorted(zip(X, weights)))
        # https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median
        totals = cumulative_sum(weights)
        target = totals[-1]*0.5
        i = bisect.bisect(totals, target)
        # TODO: interpolation
        return X[i]
    X.sort()
    if len(X) % 2 == 0:
        return (X[len(X)//2] + X[len(X)//2 - 1])/2
    else:
        return X[len(X)//2]

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
