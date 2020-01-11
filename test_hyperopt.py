from hyperopt import hp, tpe, fmin, Trials
import pickle

def q(args):
    x, y = args
    return x ** 2 + y ** 2

trials = Trials()
best = fmin(fn=q,
    space=[hp.uniform('x', -10, 10), 0],
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
print(best)
with open('testtest', 'wb') as f:
    pickle.dump(trials, f)