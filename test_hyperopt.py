from hyperopt import hp, tpe, fmin, Trials
import hyperopt
import pickle

def q(args):
    x, y = args
    ret = {'status': hyperopt.STATUS_OK, 'loss': x ** 2 + y ** 2, 'info': [x, y]}
    return ret


trials = Trials()
best = fmin(fn=q,
    space=[hp.uniform('x', -10, 10), 0],
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
print(best)
print(trials.results)
with open('testtest', 'wb') as f:
    pickle.dump(trials, f)