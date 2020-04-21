# coding=utf-8
import math
import os
import sys
import pickle
import numpy as np
import scipy.io as sio
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def filter_alias(name):
    if not name.startswith('alias'):
        return False
    else:
        return True


def get_prefix(name):
    return name[:name.find('.')]


def get_reuse_times(name):
    return int(name[name.find('_') + 1:])


def plot():
    mat_path = os.path.join('train', 'mat_percent10_topic1024_seed2019')
    # mat_name = ['alias_724', 'alias_824', 'alias_924', 'alias_1024', 'alias_1124', 'alias_1224', 'alias_1324', 'alias_1424']
    # mat_name = ['alias_8', 'alias_16', 'alias_32', 'alias_64', 'alias_128', 'alias_256', 'alias_512', 'alias_724','alias_824', 'alias_1024', 'alias_2048', 'alias_4096']
    # mat_name = ['alias_4', 'alias_8', 'alias_16', 'alias_32', 'alias_64', 'alias_88', 'alias_98', 'alias_108', 'alias_118', 'alias_123',
    #            'alias_128', 'alias_133', 'alias_138', 'alias_148', 'alias_158', 'alias_256', 'alias_512', 'alias_1024', 'alias_2048',
    #            'alias_4096', 'alias_8192', 'alias_16384', 'alias_32768']
    # 'alias_8192', 'alias_16384', 'alias_32768', 'alias_65536', 'alias_131072']
    mat_name = []

    if mat_name == []:
        mat_name = os.listdir(mat_path)
        mat_name = filter(filter_alias, mat_name)
        mat_name = list(map(get_prefix, mat_name))
        mat_name.sort(key=get_reuse_times)
        print(mat_name)

    reuse_times_list = []
    wall_time_list = []
    lim = 0
    for name in mat_name:
        path = os.path.join(mat_path, name)
        res_dict = sio.loadmat(path)
        log_likelihood = res_dict[name + '_like'][0]
        lim = max(max(log_likelihood), lim) if lim != 0 else max(log_likelihood)
        time = res_dict[name + '_time'][0]
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

    lim = 1.02 * lim
    print(lim)
    plt.figure(2)
    for name in mat_name:
        times = int(name.split('_')[1])
        wall_time = 0
        path = os.path.join(mat_path, name)
        res_dict = sio.loadmat(path)
        log_likelihood = res_dict[name + '_like'][0]
        time = res_dict[name + '_time'][0]
        time_cumul = time.copy()
        for i in range(1, len(time_cumul)):
            time_cumul[i] = time_cumul[i - 1] + time_cumul[i]
            if log_likelihood[i - 1] >= lim:
                wall_time = time_cumul[i]
                break
        reuse_times_list.append(times)
        wall_time_list.append(wall_time)
        plt.text(times, wall_time, str(times))
    plt.xlabel('reuse times')
    plt.ylabel('wall-clock time')
    plt.semilogx(reuse_times_list, wall_time_list, 'x--')
    plt.show()


def quadratic(params, x):
    """
    对e^x的二次函数
    :param params: 二次函数的参数
    :param x: 横坐标值
    :return:
    """
    a, b, c = params
    return a * x * x + b * x + c


def error(params, x, y):
    """
    误差函数
    :param params: 二次函数的参数
    :param x: 横坐标值
    :param y: 实际的纵坐标值
    :return:
    """
    return quadratic(params, x) - y


def plot_tpe(separate):
    """
    TPE 的 plot, 用于读入 trials.pk, 绘制walltime-复用次数图像
    param: separate: 是否将repeat_times>1的数据分开绘制，而不是以平均值绘制
    """
    pk_dirs = [os.path.join('train', 'mat_percent10_topic64_seed2019')]
    for pk_dir in pk_dirs:
        with open(os.path.join(pk_dir, 'trials.pk'), 'rb') as f:
            trials = pickle.load(f)

            if not separate:
                reuse_times_list = trials.idxs_vals[1]['reuse']
                wall_time_list = list(map(lambda x: x['loss'], trials.results))
                data = list(zip(reuse_times_list, wall_time_list))
                data.sort(key=lambda x: x[0])
                reuse_times_list = list(map(lambda x: x[0], data))
                wall_time_list = list(map(lambda x: x[1], data))
                log_reuse_times_list = list(map(lambda x: math.log10(x), reuse_times_list))  # 横坐标的对数值，用于拟合
                plt.figure()
                plt.xlabel('reuse times')
                plt.ylabel('wall-clock time')
                l = len(wall_time_list)
                params = leastsq(error, np.array([2, -1, 1]), args=(np.array(log_reuse_times_list), np.array(wall_time_list)))
                fit = []
                err = 0  # 计算均方根误差
                for ind, reuse_time in enumerate(reuse_times_list):
                    y = quadratic(params=params[0], x=math.log10(reuse_time))
                    fit.append(y)
                    err += (y-wall_time_list[ind]) * (y-wall_time_list[ind])
                err = math.sqrt(err/l)
                plt.semilogx(reuse_times_list, wall_time_list, 'x--')
                plt.semilogx(reuse_times_list, fit, 'x-')
                best_reuse_time = 10**(-params[0][1] / (2*params[0][0]))
                print(best_reuse_time)
                print(err)
            else:
                print(trials.results)
                num = len(trials.results[0]['separate_losses'])
                for i in range(num):
                    reuse_times_list = trials.idxs_vals[1]['reuse']
                    wall_time_list = list(map(lambda x: x['separate_losses'][i], trials.results))
                    data = list(zip(reuse_times_list, wall_time_list))
                    data.sort(key=lambda x: x[0])
                    reuse_times_list = list(map(lambda x: x[0], data))
                    wall_time_list = list(map(lambda x: x[1], data))
                    # plt.figure()
                    plt.xlabel('reuse times')
                    plt.ylabel('wall-clock time')
                    plt.semilogx(reuse_times_list, wall_time_list, 'x--')
            print(trials.best_trial)
            plt.show()


if __name__ == '__main__':
    plot_tpe(False)
