# coding=utf-8
import os
import scipy.io as sio
from matplotlib import pyplot as plt

mat_path = 'mat/'
# mat_name = ['alias_724', 'alias_824', 'alias_924', 'alias_1024', 'alias_1124', 'alias_1224', 'alias_1324', 'alias_1424']
mat_name = ['alias_8', 'alias_16', 'alias_32', 'alias_64', 'alias_128', 'alias_256', 'alias_512', 'alias_724','alias_824', 'alias_1024', 'alias_2048', 'alias_4096']
# 'alias_8192', 'alias_16384', 'alias_32768', 'alias_65536', 'alias_131072']
for name in mat_name:
    path = os.path.join(mat_path, name)
    res_dict = sio.loadmat(path)
    log_likelihood = res_dict[name+'_like'][0]
    time = res_dict[name+'_time'][0]
    time_cumul = time.copy()
    for i in range(1, len(time_cumul)):
        time_cumul[i] = time_cumul[i - 1] + time_cumul[i]
    plt.figure(0)
    plt.plot(range(1, len(log_likelihood) + 1), log_likelihood)
    plt.figure(1)
    plt.plot(time_cumul, log_likelihood)

plt.figure(0)
plt.legend(mat_name)
plt.xlabel('Number of iterations')
plt.ylabel('log likelihood')
plt.figure(1)
plt.legend(mat_name)
plt.xlabel('seconds elapsed')
plt.ylabel('log likelihood')
plt.show()
