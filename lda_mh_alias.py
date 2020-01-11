# coding=utf-8
import random
import numpy as np
import math
import pickle
import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from hyperopt import fmin, tpe, hp, Trials

class SortedSparseHistogram:
    """
        This object is used to store and
        maintain a sparse data structure
        of lda model.
    """

    def __init__(self):
        self.topic_count = []
        self.topic_index = []

    def lookup(self, topic_id):
        for index, topic in enumerate(self.topic_index):
            if topic == topic_id:
                return self.topic_count[index]
        return 0

    def output_count(self):
        return self.topic_count, self.topic_index

    def increase_count(self, topic_id, count=1):
        index = -1
        for indextemp, topic in enumerate(self.topic_index):
            if topic == topic_id:
                index = indextemp
                self.topic_count[index] += count
                break

        if index == -1:
            self.topic_count.append(count)
            self.topic_index.append(topic_id)
            index = len(self.topic_index) - 1

        indextemp = self.topic_index[index]
        counttemp = self.topic_count[index]
        while index > 0 and counttemp > self.topic_count[index - 1]:
            self.topic_index[index] = self.topic_index[index - 1]
            self.topic_count[index] = self.topic_count[index - 1]
            index -= 1
        self.topic_index[index] = indextemp
        self.topic_count[index] = counttemp
        return counttemp

    def decrease_count(self, topic_id, count=1):
        index = -1
        for indextemp, topic in enumerate(self.topic_index):
            if topic == topic_id:
                index = indextemp
                self.topic_count[index] -= count
                break

        indextemp = self.topic_index[index]
        counttemp = self.topic_count[index]
        if counttemp <= 0:
            del self.topic_index[index]
            del self.topic_count[index]
        else:
            while index < len(self.topic_index) - 1 and counttemp < self.topic_count[index + 1]:
                self.topic_count[index] = self.topic_count[index + 1]
                self.topic_index[index] = self.topic_index[index + 1]
                index += 1
            self.topic_index[index] = indextemp
            self.topic_count[index] = counttemp
        return counttemp


class AliasSamples:
    """
        This object is used to generate and
        store alias table of given distribution.
        This technique can reduce the sampling
        complexity to O(1).
    """

    def __init__(self):
        self.prob = []
        self.P = 0
        self.sample = []

    def isEmpty(self):
        return len(self.sample) == 0

    def getSample(self):
        return self.sample.pop(), self.prob, self.P

    def genSamples(self, prob, num):
        self.P = sum(prob)
        self.prob = prob / self.P
        L = []
        H = []
        A = []
        probTemp = self.prob * len(self.prob)  # debug: Should be len(self.prob) rather than num.
        for i, p in enumerate(probTemp):
            if p > 1:
                H.append((i, p))
            elif p == 1:
                A.append((i, -1, p))
            else:
                L.append((i, p))
        while len(L) > 0:
            ph = H.pop()
            pl = L.pop()
            A.append((pl[0], ph[0], pl[1]))
            p = ph[1] + pl[1] - 1
            if abs(p - 1) < 0.00000000001:
                A.append((ph[0], -1, 1))
            elif p > 1:
                H.append((ph[0], p))
            else:
                L.append((ph[0], p))

        for i in range(num):
            item = random.choice(A)
            p = random.random()
            if p < item[2]:
                self.sample.append(item[0])
            else:
                self.sample.append(item[1])
        return self.sample.pop(), self.prob, self.P


class Lda_MH_Alias:
    """ An efficient lda model based on Metropolis-Hasting-alias sampling."""

    def __init__(self):
        """Initiate the parameters of lda model.

        Parameters
        ----------------
        None

        Returns
        ----------
        None
         """
        self.reuse_num = 0  # 复用次数
        self.topic_num = 0  # 主题数目
        self.alpha = []  # 文档-主题的Dirichlet参数
        self.beta = []  # 主题-单词的Dirichlet参数
        self.words = []  # 所有的单词，以标号形式表示
        self.docs = []  # 每个单词对应的文档的标号
        self.vocabulary = []  # 标号对应的单词字符串
        self.topics = []  # 每个单词对应的主题的标号
        self.doc_num = 0  # 总文档数
        self.word_num = 0  # 总单词数（重复出现计多次）
        self.voc_num = 0  # 总词汇数（重复出现计一次）
        self.theta = []
        self.phi = []
        self.title = []

    def save_model(self, filename='model_file'):
        """Save a trained model into a pickle file.

        Parameters
        ----------------
        filename: String, (default = 'model_file')
            the file path where the model will be saved.

        Returns
        ----------
        None
        """
        f = open(filename, 'wb')
        pickle.dump(self.reuse_num, f)
        pickle.dump(self.topic_num, f)
        pickle.dump(self.beta, f)
        pickle.dump(self.alpha, f)
        pickle.dump(self.words, f)
        pickle.dump(self.docs, f)
        pickle.dump(self.vocabulary, f)
        pickle.dump(self.topics, f)
        pickle.dump(self.doc_num, f)
        pickle.dump(self.word_num, f)
        pickle.dump(self.voc_num, f)
        pickle.dump(self.theta, f)
        pickle.dump(self.phi, f)
        pickle.dump(self.title, f)
        f.close()

    def load_model(self, filename='model_file'):
        """Load in an existing model from a pickle file.

        Parameters
        ----------------
        filename: String, (default = 'model_file')
            the file path of the existing lda model.

        Returns
        ----------
        None
         """
        if not os.path.exists(filename):
            debug('Model file does not exist!')
        else:
            f = open(filename, 'rb')
            self.reuse_num = pickle.load(f)
            self.topic_num = pickle.load(f)
            self.beta = pickle.load(f)
            self.alpha = pickle.load(f)
            self.words = pickle.load(f)
            self.docs = pickle.load(f)
            self.vocabulary = pickle.load(f)
            self.topics = pickle.load(f)
            self.doc_num = pickle.load(f)
            self.word_num = pickle.load(f)
            self.voc_num = pickle.load(f)
            self.theta = pickle.load(f)
            self.phi = pickle.load(f)
            self.title = pickle.load(f)
            f.close()

    def load_data_formal(self, filename, percentage):
        """Load the training data.
        :param
        filename: String
            the file path of the training data.
        percentage: int
            the percentage of original data to be used. Note that the percentage needs to be divided by 100.

        :return:
        None

        Notes:
        The training data is formatted as the description in the file named readme.txt, which is shown as follow:
            <The number of documents>
            <The number of words in the vocabulary>
            <The total number of words>
            <The document ID of the first word> <The word ID of the first word> <The number of occurrences of the first word>
            ...

        Note that the ID ranges from 1 to the maximum value in the original file, so it needs to minus one.
        And the vocabulary is stored in the text file whose prefix name is vocab.
        """
        with open(filename, 'r') as f:
            self.words = []
            self.docs = []
            self.title = []
            self.vocabulary = []
            self.doc_num = int(f.readline())
            _ = f.readline()
            _ = f.readline()
            self.voc_num = 0
            self.word_num = 0
            # self.vocabulary = range(0, self.voc_num)
            self.doc_num = int(self.doc_num * percentage / 100)
            for line in f:
                items = list(map(int, line.strip().split(' ')))
                if items[0] > self.doc_num:
                    break
                doc = items[0] - 1
                word = items[1] - 1
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
                    self.words.extend([len(self.vocabulary) - 1] * items[2])
                else:
                    self.words.extend([self.vocabulary.index(word)] * items[2])
                self.word_num += items[2]
                self.docs.extend([doc] * items[2])
            self.voc_num = len(self.vocabulary)
            debug('Document number:\t' + str(self.doc_num) + '\tWord number:\t' + str(
                self.word_num) + '\tVocabulary number:\t' + str(self.voc_num))

    def load_data(self, filename='data'):
        """Load in the training data.
        Parameters
        ----------------
        filename:String
            the file path of the training data.

        Returns
        ----------
        None

        Notes
        --------
        The training data should be formatted as follow:
            <Number of the documents>
            <#1 word of #1 document> <#2 word of #1 document> <#3 word of #1 document> ...
            <#1 word of #2 document>...
            ...
        Where the first line is the total number of documents. After that, each line corresponds to one document,
        and the words of each document are separated by one whitespace.
        """
        f = open(filename)
        self.words = []
        self.docs = []
        self.vocabulary = []
        self.title = []

        count = 0
        doc_id = 0
        for line in f:
            line = line.strip()
            linetemp = line.split('\t')
            self.title.append(linetemp[0])
            for item in linetemp[1].split():
                count += 1
                if item not in self.vocabulary:
                    self.vocabulary.append(item)
                    self.words.append(len(self.vocabulary) - 1)
                else:
                    self.words.append(self.vocabulary.index(item))

                self.docs.append(doc_id)
            doc_id += 1

        self.voc_num = len(self.vocabulary)
        self.doc_num = doc_id
        self.word_num = count
        debug('Document number:\t' + str(self.doc_num) + '\tWord number:\t' + str(
            self.word_num) + '\tVocabulary number:\t' + str(self.voc_num))
        f.close()

    def cal_log_likelihood(self, phi, theta):
        """Calculate the log likelihood of the current model.

        Parameters
        ----------------
        phi: topic_word distribution for every topic
        theta: doc_topic distribution for every doc

        Returns
        ----------
        log_likelihood: float
            The log likelihood value of the current lda model.
         """
        log_likelihood = 0
        prob = np.dot(theta, phi)
        for i in range(self.word_num):
            doc_id = self.docs[i]
            word_id = self.words[i]
            log_likelihood += math.log(prob[doc_id, word_id])
        debug('\tLog likelihood:\t' + str(log_likelihood))
        return log_likelihood

    def sample(self, p):
        """Sample out a new topic based on the given probability distribution.
        This is a sub-function of the training function.

        Parameters
        ----------------
        p: array, float
            The given probability distribution on each topic.

        Returns
        ----------
        statu: int
            The index of the sampled out topic.
         """
        p_cumulative = [p[0]]
        length = len(p)
        for i in range(1, length):
            p_cumulative.append(p_cumulative[-1] + p[i])
        val = random.random() * p_cumulative[-1]
        for statu in range(length):
            if val < p_cumulative[statu]:
                break
        return statu

    def print_word_feature(self, topn=15, filename='topic_feature'):
        """Print out several most important words of each topic (i.e. the words
        with highest probability that belong to this topic) into a specific file.
        These words can be viewed as the feature of topics.

        Parameters
        ----------------
        topn: int, (default = 15)
            The number of the most important words of each topic.

        filename: String, (default = 'topic_feature')
            The file path where the feature of each topic will be stored.

        Returns
        ----------
        None
        """
        f = open(filename, 'w')
        phitemp = self.phi.copy()
        for topic_id in range(self.topic_num):
            f.write('Topic ' + str(topic_id) + ':\n')
            for i in range(topn):
                index = np.argmax(phitemp[topic_id, :])
                f.write(self.vocabulary[index] + ' : ' + str(phitemp[topic_id, index]) + '\n')
                phitemp[topic_id, index] = 0
        f.close()

    def print_topic_feature(self, filename='document_feature'):
        """Print out the topic feature of each document.
        (i.e. the probability feature of each document.)

        Parameters
        ----------------
        filename: String, (default = 'topic_feature')
            The file path where the feature of each document will be stored.

        Returns
        ----------
        None
         """
        f = open(filename, 'w')
        for doc_id in range(self.doc_num):
            f.write(self.title[doc_id] + ':')
            for item in self.theta[doc_id, :]:
                f.write(str(item) + ' ')
            f.write('\n')
        f.close()

    def train(self, reuse_num, topic_num, max_iter_num=100, valid_sample_num=10, alpha=0.01, beta=0.1, threshold=-1):
        """Train a lda model based on the given parameters. The training process
        is based on Metropolis-Hasting-alias sampling technique. The last several iterations( the
        sampling is believed to be converged in these iterations) of the
        sampling are recorded to estimate the parameters of the model.

        Parameters
        ----------------
        reuse_num: int
            The number of reuse of Alias Table.

        topic_num: int, (default = 3)
            The given latent topic number of lda model.

        max_iter_num: int, (default = 100)
            The total iteration number of the training.

        valid_sample_num: int, (default = 10)
            The number of the last several iterations that is considered as the
            valid sample(i.e. the Gibbs sampling is converged). These samples
            are used to estimate the parameters of lda model.

        alpha: float, (default = 0.01)
            The hyper parameters of the dirichlet distribution of document-topic model.

        beta: float, (default = 0.1)
            The hyper parameters of the dirichlet distribution of topic-word model.

        threshold: float, (default = -1)
            If threshold is not -1, the training process will stop once the loss value is
            greater than threshold.
        Returns
        ----------
        t: array, float
            The consumed time of each iterations. (The unit is second).

        log_likelihood: array, float
            The log_likelihood of the topic model at every 10 iterations.
         """
        # Initiate the parameters of training model
        self.topic_num = topic_num
        self.reuse_num = reuse_num
        self.alpha = np.ones((topic_num, 1))[:, 0] * alpha
        self.beta = np.ones((self.voc_num, 1))[:, 0] * beta
        topic_pool = range(self.topic_num)

        # Initiate the topic of each word by randomly choosing all the topics
        self.topics = []
        for i in range(self.word_num):
            self.topics.append(random.choice(topic_pool))

        # Initiate the statistics that are used to calculate the possibility of Gibbs sampling
        topic_sum = np.arange(self.topic_num) * 0
        topic_word_count = np.zeros((self.topic_num, self.voc_num))
        doc_topic_count = [SortedSparseHistogram() for i in range(self.doc_num)]
        alias_sample = [AliasSamples() for i in range(self.voc_num)]
        for doc_id, topic_id, word_id in zip(self.docs, self.topics, self.words):
            doc_topic_count[doc_id].increase_count(topic_id)
            topic_word_count[topic_id, word_id] += 1
            topic_sum[topic_id] += 1

        alpha_sum = sum(self.alpha)
        beta_sum = sum(self.beta)

        # Initiate the statistics of the valid sample that are used estimate parameters
        sample_topic_word_count = np.zeros((self.topic_num, self.voc_num))
        sample_doc_topic_count = np.zeros((self.doc_num, self.topic_num))

        iter_num = 0
        valid_thresh = max_iter_num - valid_sample_num
        t = []
        log_likelihood = []
        p_temp = np.arange(self.topic_num) * 0
        count_temp = np.arange(self.topic_num) * 0

        # MH_alias sampling the topic of each word, until the iteration number is met
        while iter_num != max_iter_num:
            iter_num += 1

            t1 = time.process_time()
            for i in range(self.word_num):
                doc_id = self.docs[i]
                word_id = self.words[i]
                topic_id = self.topics[i]

                doc_topic_count[doc_id].decrease_count(topic_id)
                topic_word_count[topic_id, word_id] -= 1
                topic_sum[topic_id] -= 1
                # 对每个词汇(voc)采样生成若干样本，样本数用完之后再重新使用 Alias 算法
                if alias_sample[word_id].isEmpty():
                    prob = alpha * (topic_word_count[:, word_id] + beta) / (topic_sum + beta_sum)
                    Q_sample, q_prob, Q = alias_sample[word_id].genSamples(prob, self.reuse_num)
                else:
                    Q_sample, q_prob, Q = alias_sample[word_id].getSample()

                topic_count, topic_index = doc_topic_count[doc_id].output_count()
                p_prob = topic_count * (topic_word_count[topic_index, word_id] + beta) / (
                        topic_sum[topic_index] + beta_sum)
                P = sum(p_prob)
                p_prob = p_prob / P
                if random.random() < (P / (P + Q)):
                    topic_id_proposed = topic_index[self.sample(p_prob)]
                else:
                    topic_id_proposed = Q_sample

                p_temp[topic_index] = p_prob
                count_temp[topic_index] = topic_count

                accept_rate = \
                    (P * p_temp[topic_id] + Q * q_prob[topic_id]) / (
                            P * p_temp[topic_id_proposed] + Q * q_prob[topic_id_proposed]) \
                    * (topic_sum[topic_id] + beta_sum) / (topic_sum[topic_id_proposed] + beta_sum) \
                    * (topic_word_count[topic_id_proposed, word_id] + beta) / (
                            topic_word_count[topic_id, word_id] + beta) \
                    * (count_temp[topic_id_proposed] + alpha) / (count_temp[topic_id] + alpha)

                accept_rate = min(accept_rate, 1)

                if random.random() < accept_rate:
                    topic_id = topic_id_proposed

                p_temp[topic_index] = 0
                count_temp[topic_index] = 0

                self.topics[i] = topic_id
                doc_topic_count[doc_id].increase_count(topic_id)
                topic_word_count[topic_id, word_id] += 1
                topic_sum[topic_id] += 1

            t2 = time.process_time()
            t.append(t2 - t1)
            debug('iteration %d: consumed %.2f' % (iter_num, t[-1]))
            if iter_num % 1 == 0:
                phi = np.zeros((self.topic_num, self.voc_num))
                theta = np.zeros((self.doc_num, self.topic_num))
                doc_topic = np.zeros((self.doc_num, self.topic_num))

                for doc_id, item in enumerate(doc_topic_count):
                    topic_count, topic_index = item.output_count()
                    doc_topic[doc_id, topic_index] += topic_count

                for i in range(self.topic_num):
                    phi[i, :] = (topic_word_count[i, :] + self.beta) / (sum(topic_word_count[i, :]) + beta_sum)
                for i in range(self.doc_num):
                    theta[i, :] = (doc_topic[i, :] + self.alpha) / (sum(doc_topic[i, :]) + alpha_sum)

                like = self.cal_log_likelihood(phi, theta)
                log_likelihood.append(like)
                if threshold != -1 and like >= threshold:
                    break

        #         if iter_num>valid_thresh:
        #             sample_topic_word_count += topic_word_count
        #         for doc_id, item in enumerate(doc_topic_count):
        #             topic_count, topic_index = item.output_count()
        #             sample_doc_topic_count[doc_id, topic_index] += topic_count

        # # Estimate the parameters based on the valid samples
        # self.phi = np.zeros((self.topic_num, self.voc_num))
        # self.theta = np.zeros((self.doc_num, self.topic_num))
        # for i in range(self.topic_num):
        #     self.phi[i,:] = (sample_topic_word_count[i,:] + self.beta )/(sum(sample_topic_word_count[i,:]) + beta_sum)
        # for i in range(self.doc_numF):
        #     self.theta[i,:]=(sample_doc_topic_count[i,:] + self.alpha)/(sum(sample_doc_topic_count[i,:]) + alpha_sum)

        return t, log_likelihood

    def get_wallclock(self, args):
        """
        对于单个复用次数进行测试，用于 run_auto
        :param reuse_num: 复用次数
        :param seed: 种子
        :param topic_num: 主题数
        :param threshold: 阈值，loss超过该阈值则截断
        :param log_dir: log文件存放目录
        :param train_dir: 训练文件存放目录
        :param repeat_times: 重复次数，第 i 次用的种子为 seed + i*i
        :return: 返回在 repeat_times 次测试中， loss 值到达 threshold 所需的平均时间
        """
        reuse_num, seed, topic_num, threshold, log_dir, train_dir, repeat_times = args
        reuse_num = int(reuse_num)
        topic_num = int(topic_num)
        repeat_times = int(repeat_times)
        debug('get_wallclock with %d reuse times' % reuse_num)
        p = 'alias_' + str(reuse_num)
        global log_path
        log_path = os.path.join(log_dir, 'log_' + p + '.txt')
        path = os.path.join(train_dir, p)
        log_likelihood = None
        ret = 0
        for i in range(repeat_times):
            random.seed(seed + i * i)
            t, l = self.train(reuse_num=reuse_num, topic_num=topic_num, threshold=threshold)
            ret += sum(t)
        # self.save_model('models/alias_' + str(reuse_num) + '_model')
        data = {p + '_time': t, p + '_like': log_likelihood}
        # sio.savemat(path, data) 在这里不存储log_likelihood的训练过程
        return ret/repeat_times


    def run(self, reuse_list, percentage, seed, topic_num):
        """Run the training of a series of lda models with different numbers of reuse.
        The performance of each lda model is evaluated with log likelihood and
        consumed time. The results are stored in a mat file.

        Parameters
        ----------------
        start: int, (default = 2)
            The minimum number of topic.

        end: int, (default = 6)
            The maximu number of topic.

        Returns
        ----------
        None
        """
        train_dir = os.path.join('train', 'mat_percent%d_topic%d_seed%d' % (percentage, topic_num, seed))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        log_dir = os.path.join(train_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        global log_path
        log_path = os.path.join(log_dir, 'settings.txt')
        self.load_data_formal(filename='data/docword.enron.txt/docword.enron.txt', percentage=percentage)
        for i in reuse_list:
            random.seed(seed)  # fix the seed.
            debug('training model with %d reuse times' % i)
            p = 'alias_' + str(i)
            log_path = os.path.join(log_dir, 'log_' + p + '.txt')
            path = os.path.join(train_dir, p)
            t, log_likelihood = self.train(reuse_num=i, topic_num=topic_num)
            self.save_model('models/alias_' + str(i) + '_model')
            data = {p + '_time': t, p + '_like': log_likelihood}
            sio.savemat(path, data)

    def run_auto(self, percentage, seed, topic_num, threshold, repeat_times, max_evals):
        """
        Run model without reuse_list, but with the selection algorithm Auto-WEKA with TPE.
        :param percentage: 只使用数据的percentage%部分
        :param seed: 用于固定随机种子
        :param topic_num: 主题数
        :param threshold: 阈值，用于计算wallclock——训练直到损失值大于wallclock所需时间
        :param repeat_times: 对于每一个复用次数，重复进行的测试次数
        :param max_evals: 最大迭代次数
        :return: 
        """
        train_dir = os.path.join('train', 'mat_percent%d_topic%d_seed%d' % (percentage, topic_num, seed))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        log_dir = os.path.join(train_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        global log_path
        log_path = os.path.join(log_dir, 'settings.txt')
        self.load_data_formal(filename='data/docword.enron.txt/docword.enron.txt', percentage=percentage)
        trials = Trials()
        best = fmin(fn=self.get_wallclock,
                    space=[hp.quniform('reuse', 16, 4096, 1), seed, topic_num, threshold,
                           log_dir, train_dir, repeat_times],  # TODO: loguniform
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        #reuse_num, seed, topic_num, threshold, log_dir, train_dir
        with open(os.path.join(train_dir, 'trials.pk'), 'wb') as f:
            pickle.dump(trials, f)

log_path = ''

def debug(s):
    global log_path
    print(s)
    with open(log_path, 'a') as f:
        print(s, file=f)

# 注意percentage必须是整数
model = Lda_MH_Alias()
# model.run([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], percentage=10, seed=2019, topic_num=256)
# model.run([138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 268, 278, 288, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480], percentage=10, seed=8374)
# model.run([724, 824, 924, 1024, 1124, 1224, 1324, 1424])

# TODO: pay attention to the threshold.
model.run_auto(percentage=10, seed=2019, topic_num=256, threshold=-2800000, repeat_times=3, max_evals=30)
