from __future__ import print_function
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn



### 从文件获取数据
# my_matrix = np.loadtxt(open("data2.csv", "rb"), dtype=np.float, delimiter=",", skiprows=1)
import pandas as pd
import os

os.chdir('C:\\Users\\Qiao\\Desktop\\TensorDemo-master\\train_set_process_by_column_add_xy_distance_level_relative_degree')
file_chdir = os.getcwd()
filecsv_list = []
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            filecsv_list.append(file)
FileSize=len(filecsv_list)


def get_distance(data):
    data['distance']  = np.sqrt(pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)) * 5
    #data['distance']  = pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)
    return data

def get_relative_height(data):
    #data['distance']  = math.sqrt(pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)) * 5
    data['relative_height']  = np.fabs(data['Cell Altitude'] + data["Height"] - (math.tan(math.radians(data['Electrical Downtilt'] + data['Mechanical Downtilt'])) * data['distance'] + data['Altitude']) )
    return data

def get_x_distance(data):
    data['x_distance']  = np.sqrt(pow(data['Cell X'] - data['X'], 2)) * 5
    return data

def get_y_distance(data):
    data['y_distance']  = np.sqrt(pow(data["Cell Y"] - data['Y'], 2)) * 5
    return data

def count_level_relative_degree(cellx,celly,x,y):
    degree = math.degrees(math.atan( (celly - y)/(cellx - x) ))
    if degree < 0:
        degree = 180- math.fabs(degree)
    return degree

def get_level_relative_degree(data):
    # 计算水平相对角度
    # 先判断是否为特殊的点  在x轴上或者在y轴上
    cellx = data['Cell X']
    celly = data['Cell Y']
    x = data['X']
    y = data['Y']
    Azimuth = data['Azimuth']
    jiaodu = 0
    if cellx == x:
        if celly > 0:
            jiaodu = math.fabs(90 - Azimuth)
        else:
            jiaodu = math.fabs(Azimuth - 270)
    if celly == y:
        if x > 0:
            jiaodu = math.fabs(90 - Azimuth)
        else:
            jiaodu = math.fabs(Azimuth - 270)
    # 在象限内
    if cellx < x and y > celly:  # 1
        jiaodu = 90 - count_level_relative_degree(cellx, celly, x, y) - Azimuth
    elif cellx > x and y > celly:  # 2
        jiaodu = 360 - count_level_relative_degree(cellx, celly, x, y) + 90 - Azimuth
    elif cellx > x and y < celly:  # 3
        jiaodu = 90 - count_level_relative_degree(cellx, celly, x, y) + 180 - Azimuth
    elif cellx < x and y < celly:  # 4
        jiaodu = 180 - count_level_relative_degree(cellx, celly, x, y) + 90 - Azimuth
    jiaodu = math.fabs(jiaodu)
    if jiaodu <= 180:
        data['level_relative_degree'] =  math.fabs(jiaodu)
    else:
        data['level_relative_degree'] = math.fabs(jiaodu - 180)
    return data


def getFileFuture(FileName):

    data = pd.read_csv(FileName, header=0, sep=None)
    return  data



def getFileData(batch):
    x_input_data = []
    y_input_data = []

    num=1000*batch
    for file_content in filecsv_list:
        print("第"+str(num)+"个文件！")
        pb_data = getFileFuture(file_content)
        input_x_data = np.array(pb_data.get_values()[:, 0:24], dtype=np.float32)
        input_y_data = np.array(pb_data.get_values()[:, 24:25], dtype=np.float32)
        x_input_data.extend(input_x_data)
        y_input_data.extend(input_y_data)
        num=num+1
        if num>=1000*batch+1000:
            break

    x_input_data = np.array(x_input_data)
    y_input_data = np.array(y_input_data)
    # x_input_data=np.delete(x_input_data,[0,1,2,5,6,7,10,11,12,13,15,16],axis=1)
    x_input_data = np.delete(x_input_data, [0, 1, 2, 5, 6, 10, 12, 13, 15], axis=1)
    # x_input_data = np.delete(x_input_data, [0, 1, 2, 5, 6, 7, 10, 12, 13], axis=1)
    tezheng =x_input_data
    sale = y_input_data
    X_train = Normalizer().fit_transform(tezheng)   ## 多维特征
    # 假设最大值为-70 最小值为-110  归一化操作
    MaxValue=-70
    MinValue=-110
    rang=MaxValue-MinValue
    sale=np.array(sale)
    sale=(sale-MinValue)/rang
    y_train = sale.reshape((-1,1))        ## 结果，从一维转为二维
    return X_train,y_train



def get_Batch(X_train,y_train,i):
    return X_train[(i-1)*1000:i*1000,:], y_train[(i-1)*1000:i*1000]

#### 开始进行图的构建

## 特征与结果的替代符，声明类型，维度 ，name是用来生成模型之后，使用模型的时候调用用的
inputX = tf.placeholder(shape=[None, 15], dtype=tf.float32, name="myInput")
y_true = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y_true")


### 第一层，一个隐藏层 开始
## shape的第一维就是特征的数量，第二维是给下一层的输出个数,  底下的矩阵相乘实现的该转
Weights1 = tf.Variable(tf.random_normal(shape=[15, 3000]), name="weights1")  ## 权重
biases1 = tf.Variable(tf.zeros(shape=[1, 3000]) + 0.1, name="biases1")       ## 偏置
## matmul矩阵相乘，nn.dropout 丢弃部分不靠谱数据
Wx_plus_b1 = tf.matmul(inputX, Weights1)
Wx_plus_b1 = tf.add(Wx_plus_b1, biases1)
Wx_plus_b1 = tf.nn.dropout(Wx_plus_b1, keep_prob=0.9)
## 将结果曲线化，通常说非线性化
l1 = tf.nn.sigmoid(Wx_plus_b1, name="l1")
### 第一层结束

### 第2层， 开始
Weights2 = tf.Variable(tf.random_normal(shape=[3000, 3000]), name="weights2")  ## 权重
biases2 = tf.Variable(tf.zeros(shape=[1, 3000]) + 0.1, name="biases2")       ## 偏置
## matmul矩阵相乘，nn.dropout 丢弃部分不靠谱数据
Wx_plus_b2 = tf.matmul(l1, Weights2)
Wx_plus_b2 = tf.add(Wx_plus_b2, biases2)
Wx_plus_b2 = tf.nn.dropout(Wx_plus_b2, keep_prob=0.9)
## 将结果曲线化，通常说非线性化
l2 = tf.nn.sigmoid(Wx_plus_b2, name="l1")
### 第一层结束

### 第3层开始，即输出层
## 上一层的10，转为1，即输出销售量
Weights3 = tf.Variable(tf.random_normal(shape=[3000, 1]), name="weights3")   ## 权重
biases3 = tf.Variable(tf.zeros(shape=[1, 1]) + 0.1, name="biases3")        ## 偏置

## matmul矩阵相乘 ,l1 为上一层的结果
Wx_plus_b3 = tf.matmul(l2, Weights3)
prediction = tf.add(Wx_plus_b3, biases3, name="myOutput")  ## pred用于之后使用model时进行恢复

## 这里使用的这个方法还做了一个第一维结果行差别的求和，reduction_indices=1，实际这个例子每行只有一个结果,使用 loss = tf.reduce_sum(tf.square(y_true - prediction)) 即可
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

## 训练的operator，AdamOptimizer反正说是最好的训练器, 训练速率0.01
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

batchsize=1000

def TrueOrFalse(x):
    return x<-103
def compute_pcrr(y_pred,y_true):
    t = -103
    y_pred=np.array(y_pred).reshape(-1)
    y_true=np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred) * (110 - 70) + (-110)
    y_true = np.array(y_true) * (110 - 70) + (-110)
    num =0
    tp=0
    fp=0
    fn=0
    for item in y_pred:
        if y_true[num]<t and y_pred[num]<t:
            tp=tp+1
        elif y_true[num] >=t and y_pred[num]<t:
            fp=fp+1
        elif y_true[num]<t and y_pred[num]>=t:
            fn=fn+1
        num=num+1
    if (tp + fp)==0:
        return -1
    if (tp + fn)==0:
        return -1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision+recall)==0:
        return -1
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr

### 开始执行
with tf.Session() as sess:

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)     # 初始化saver，用于保存模型

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()  # 合并所有的summary
    writer = tf.summary.FileWriter("logs/", sess.graph)
    # 初始化全部变量
    sess.run(init)                                                     # 初始化全部变量

    ## 要给模型进行训练的数据，只有placeholder类型的需要传进去数据


    for j in range(4):
        print("EPOCH",str(j))
        for i in range(int(FileSize/1000)):
            X_train,y_train =getFileData(i)
            for k in  range(int(len(X_train)/1000)):
                new_x_batch,new_y_batch=get_Batch(X_train,y_train,k)
                feed_dict_train = {inputX: new_x_batch, y_true: new_y_batch}
                _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)  # 训练，注：loss没有训练，只是走到loss，返回值，走到train_op才会训练
                feed_dict_trains = {inputX: new_x_batch}
                print("acc", compute_pcrr(sess.run([prediction], feed_dict=feed_dict_trains), new_y_batch))
                print("Epoch%d:步数:%d:loss:%.5f" % (j, k, _loss))

    # 保存模型
    # saver.save(sess=sess, save_path="nn_boston_model/variables.model", global_step=10)
    tf.saved_model.simple_save(sess, "./model", inputs={"myInput": inputX}, outputs={"myOutput": prediction})


