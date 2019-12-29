from hyperopt import hp, tpe, fmin

def q(args):
    x, y = args
    return x ** 2 + y ** 2

best = fmin(fn=q,
    space=[hp.uniform('x', -10, 10), 0],
    algo=tpe.suggest,
    max_evals=1000)
print(best)