# coding=utf-8
from __future__ import absolute_import, division, print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
import pandas as pd
from sklearn.metrics import accuracy_score
import gflags
import pickle
from base.portfolio import LONG, NONE, SHORT
from base.strategy import Strategy

class conf:

    field = 'close'
    seq_len = 100  # 每个input的长度
    prediction_len = 20  # 预测数据长度
    train_proportion = 0.9  # 训练数据占总数据量的比值，其余为测试数据
    normalise = False  # 数据标准化
    epochs = 1  # LSTM神经网络迭代次数
    batch = 100  # 整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
    validation_split = 0.1  # 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
    lr = 0.001  # 学习效率


# 2. LSTM策略主体
# 例如希望根据前seq_len天的收盘价预测第二天的收盘价，那么可以将data转换为(len(data)-seq_len)(seq_len+1)的数组，
# 由于LSTM神经网络接受的input为3维数组，因此最后可将input+output转化为(len(data)-seq_len)(seq_len+1)*1的数组
class LSTMStrategy(Strategy):
    predtion = None
    y_test = None
    test_datetime = None
    hold_days = conf.prediction_len

    def __init__(self, future_portfolio):
        super(LSTMStrategy, self).__init__()
        self.fp = future_portfolio
        self.actt = None

    def load_data(self,field, seq_len, prediction_len, train_proportion,
                  normalise=True):


        with open('..\data\week-20180105-5m-ohlc.pickle', 'rb') as f:
            data1 = pickle.load(f)

        data1 = pd.DataFrame(data1)

        with open('..\data\week-20180119-5m-ohlc.pickle', 'rb') as f:
            data2 = pickle.load(f)
        data2 = pd.DataFrame(data2)

        data = pd.concat([data1, data2], axis=0)

        data = pd.DataFrame(data)
        # print(data)

        data = data[:3000]
        datetime_before = list(data.index)
        datetime = []
        for index in range(len(datetime_before)-seq_len-1):
            datetime.append(datetime_before[index+seq_len-1])

        data = list(data[field])
        seq_len = seq_len + 1
        before_result = []
        # 每100个close价存为一批，result加几批
        for index in range(len(data) - seq_len):
            before_result.append(data[index:index + seq_len])

        # 2319
        row = round(train_proportion * len(before_result))
        x_test_before = np.array(before_result)[int(row):,:100]
        if normalise:
            norm_result = self.normalise_windows(before_result)
        else:
            norm_result = before_result
        # (2899, 101)
        result = np.array(before_result)
        norm_result = np.array(norm_result)

        # (580,100)
        data_test = result[int(row):, :]
        test_date = datetime[int(row):]
        # datetime = datetime[int(row):]  # 681
        # 每100个datetime存为一批，存储几批
        # self.test_datetime = []
        # for index in range(len(datetime)):
        #     if index % prediction_len == 0 and index + seq_len < len(datetime) - prediction_len:
        #         self.test_datetime.append(datetime[index + seq_len])
        # print("test_datetime", self.test_datetime)
        train = norm_result[:int(row), :]
        np.random.shuffle(train)
        # 除最后一列外都作为输入
        x_train = train[:, :-1]
        # 将最后一列作为输出
        y_train = train[:, -1]
        x_test = norm_result[int(row):, :-1]
        y_test = norm_result[int(row):, -1]
        # x_train.shape=(2899,100,1)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return [x_train, y_train, x_test, y_test, data_test, x_test_before,test_date]

    def normalise_windows(self, window_data):
        # 数据规范化
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def denormalise_windows(self, normdata, data, seq_len):
        # 数据反规范化
        denormalised_data = []
        wholelen = 0
        for i, rowdata in enumerate(normdata):
            denormalise = list()
            if isinstance(rowdata, float) | isinstance(rowdata, np.float32):
                denormalise = [(rowdata + 1) * float(data[wholelen][0])]
                denormalised_data.append(denormalise)
                wholelen = wholelen + 1
            else:
                for j in range(len(rowdata)):
                    denormalise.append((float(rowdata[j]) + 1) * float(data[wholelen][0]))
                    wholelen = wholelen + 1
                denormalised_data.append(denormalise)
        return denormalised_data

    def build_model(self, layers):
        # LSTM神经网络层
        # 详细介绍请参考http://keras-cn.readthedocs.io/en/latest/
        model = Sequential()
        # input_length=100, input_dim=layers[0]
        model.add(LSTM(input_shape=(layers[1], layers[0]), units=layers[1], return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[1],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            input_dim=layers[1],
            units=layers[2]))
        model.add(Activation("linear"))

        rms = optimizers.RMSprop(lr=conf.lr, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=rms)
        start = time.time()
        print("> Compilation Time : ", time.time() - start)
        return model

    def predict_point_by_point(self, model, data):
        # 每次只预测1步长
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence_full(self, model, data, seq_len):
        # 根据训练模型和第一段用来预测的时间序列长度逐步预测整个时间序列
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            # shapes = curr_frame.shape
            # sahpe2 =  curr_frame[newaxis,:,:].shape
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [seq_len - 1], predicted[-1], axis=0)
        return predicted


    def predict_sequences_multiple(self, model, data, seq_len, prediction_len):
        # 根据训练模型和每段用来预测的时间序列长度逐步预测prediction_len长度的序列
        prediction_seqs = []  # 28*20
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]  # (100,1)
            predicted = []
            for j in range(prediction_len):
                # predict的shape是[1,100,1],predict后shape为（1,1）
                predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
                # 取每段的后一个时间序列长度,相当于输入xt和ht, curr_frame相当于新的输入
                curr_frame = curr_frame[1:]
                # insert(arr,index,value, axis) predicted相当于history,
                curr_frame = np.insert(curr_frame, [seq_len - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs


    def plot_results(self, plt, predicted_data, true_data):
        # 做图函数，用于predict_point_by_point和predict_sequence_full
        x = np.arange(0, len(predicted_data), 1)
        l1, = plt.plot(x, predicted_data, label='predictions')
        l2, = plt.plot(x, true_data, label='true_data', linewidth=.5, color=(.5, .5, .5))
        plt.ylabel("price")
        plt.legend(handles=[l1, l2, ], labels=['predictions', 'true_data'], loc='upper right')
        plt.title("true_data and predictions")


    def plot_results_multiple(self,plt, predicted_data, true_data, prediction_len):
        # 做图函数，用于predict_sequences_multiple
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data)
        plt.legend()
        # figure = plt.gcf()
        # figure.set_size_inches(20, 10)
        plt.title("true data and %s days trend"%conf.prediction_len)
        plt.show()


    def SleepFor(self, t, p):
        self.actt = t + pd.Timedelta(p)


    def train_predict(self):
        # 主程序
        # mylstm = LSTMStrategy()
        global_start_time = time.time()

        print('> Loading data... ')

        X_train, y_train, X_test, y_test, data_test, x_test_before,test_date = self.load_data(conf.field, conf.seq_len,
                                                                                      conf.prediction_len,
                                                                                      conf.train_proportion, normalise=True)

        print('> Data Loaded. Compiling...')

        model = self.build_model([1, conf.seq_len, 1])

        model.fit(
            X_train,
            y_train,
            batch_size=conf.batch,
            epochs=conf.epochs,validation_data=( X_test, y_test))

        # predictions = self.predict_sequence_full(model, X_test, conf.seq_len)
        predictions = self.predict_sequences_multiple(model, X_test, conf.seq_len, conf.prediction_len)
        # predictions = predict_sequence_full(model, X_test, conf.seq_len)
        # predictions = predict_point_by_point(model, X_test)

        if conf.normalise == False:
            predictions = self.denormalise_windows(predictions, data_test, conf.seq_len)
            y_test = self.denormalise_windows(y_test, data_test, conf.seq_len)


        y_predict = []
        y_label = []

        for i in range(len(y_test)-1):
            if y_test[i+1] > y_test[i]:
                y_label.append(1)
            else:
                y_label.append(0)
        for i in range(len(predictions)-1):
            if predictions[i+1]>predictions[i]:
                y_predict.append(1)
            else:
                y_predict.append(0)

        # accuracy = accuracy_score(y_label, y_predict)
        # print("accuracy_score", accuracy)

        print('Training duration (s) : ', time.time() - global_start_time)
        # self.plot_results_multiple(predictions, y_test, conf.prediction_len)

        # plot_results(predictions, y_test)
        self.prediction = predictions
        # self.plot_true_predict(y_test, self.prediction)
        return predictions, x_test_before,test_date,y_test

    def plot_true_predict(self, plt, predictions,y_test ):


        # plt.xlim(0, len(predictions))
        # min_price = min(np.min(np.array(y_test)), np.min(np.array(predictions)))
        # max_price = max(np.max(np.array(y_test)),np.max(np.array(predictions)))
        # plt.ylim(min_price,max_price)
        # my_x_ticks = np.arange(0,len(predictions), 0.1)
        # my_y_ticks = np.arange(min_price, max_price, 0.1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)

        x = np.arange(0, len(predictions), 1)
        l1, = plt.plot(x,predictions, label='predictions')
        l2, = plt.plot(x, y_test, label='true_data', linewidth=.5, color=(.5, .5, .5))
        plt.ylabel("price")
        plt.legend(handles=[l1, l2, ], labels=['predictions', 'true_data'], loc='upper right')
        plt.title("true_data and predictions")


    def Setup(self):
        self.AddOhlc('30min')


    def tick(self, t, price, i):

        self.fp.UpdatePrice(t, price)

        fp = self.fp

        if self.prediction[i+conf.prediction_len] >self.prediction[i]:
            if fp.direction == NONE:
                fp.AdjustPosition(LONG, 1. / 3.)
            elif fp.direction == SHORT:
                fp.AdjustPosition(NONE)
            else:
                fp.AdjustPosition(LONG, 1. / 3.)
            self.SleepFor(t, '15min')
        elif self.prediction[i+conf.prediction_len] <self.prediction[i]:
            if fp.direction == NONE:
                fp.AdjustPosition(SHORT, 1. / 3.)
            elif fp.direction == LONG:
                fp.AdjustPosition(NONE)
            else:
                fp.AdjustPosition(SHORT, 1. / 3.)
            self.SleepFor(t, '15min')


    def tick_multiple(self, t, price, i):
        self.fp.UpdatePrice(t, price)

        fp = self.fp

        if self.prediction[i][-1] > self.prediction[i][0]:
            if fp.direction == NONE:
                fp.AdjustPosition(LONG, 1. / 3.)
            elif fp.direction == SHORT:
                fp.AdjustPosition(NONE)
            else:
                fp.AdjustPosition(LONG, 1. / 3.)
            self.SleepFor(t, '15min')
        elif self.prediction[i][-1] < self.prediction[i][0]:
            if fp.direction == NONE:
                fp.AdjustPosition(SHORT, 1. / 3.)
            elif fp.direction == LONG:
                fp.AdjustPosition(NONE)
            else:
                fp.AdjustPosition(SHORT, 1. / 3.)
            self.SleepFor(t, '15min')




