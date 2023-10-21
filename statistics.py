def average(X):
    if not X:
        raise ValueError('Input must be non-empty')
    return sum(X)/len(X)

def variance(X):
    mu = average(X)
    return sum(x*x for x in X)/len(X) - mu*mu

def median(X):
    if not X:
        raise ValueError('Input must be non-empty')
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
