# -*- coding: utf-8 -*-

#LSTM 分类模型  手写数字识别


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#定义超参数
lr = 0.001   #学习率
trainning_iters = 100000   #训练的步数的上限
batch_size = 128   #批量数据

n_inputs = 28     #输入数据的列数
n_setps = 28      #输入数据的行数（步数）
n_hidden_units = 128   #神经单元数
n_clases = 10      #类别数（数据的标签数）


def RNN_LSTM(X, weights, biases):
    #原始数X的形状为三维数据[128,28,28],转换为二维数据[28,28]
    X = tf.reshape(X, [-1, n_inputs])  #[None, 28, 28]
    X_in = tf.matmul(X, weights["in"]) + biases["in"]
    X_in = tf.reshape(X_in, [-1, n_setps, n_hidden_units])   #[None, 28, 128]

    #cell ,LSTM cell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)   #初始化为零（神经单元的隐状态）

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)   #[128,28,128]
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights["out"]) + biases["out"]
    return results



if __name__ == '__main__':
    #随机种子
    tf.set_random_seed(1)
    #导入数据
    print('开始加载数据...')
    mnist = input_data.read_data_sets('./mnist', one_hot=True)
    print('数据加载完毕！')

    #定义x ,y 的初始变量（占位符）
    x = tf.placeholder(tf.float32, [None, n_setps, n_inputs])    #[None, 28, 28]
    y = tf.placeholder(tf.float32, [None, n_clases])             #[None, 10]


    #定义weights, biases的初始变量
    weights = {
        "in" : tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  #[28, 128]
        "out" : tf.Variable(tf.random_normal([n_hidden_units, n_clases]))   #[128, 10]
    }

    biases = {
        "in" : tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),   #[128, ]
        "out" : tf.Variable(tf.constant(0.1, shape=[n_clases, ]))     #[10, ]
    }


    #定义模型结构，计算cost, 和预测值， 优化器
    pred = RNN_LSTM(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  #损失值
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    #定义模型的准确率
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  #判断是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   #准确率

    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()  #初始化模型参数
        sess.run(init)

        setp = 0
        while setp * batch_size < trainning_iters:
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(batch_size, n_setps, n_inputs)   #[128, 28, 28]
            sess.run([train_op], feed_dict={x : batch_xs, y :batch_ys})

            #每训练20步打印一下准确率
            if setp % 20 == 0:
                print(setp, "准确率：%s" % sess.run(accuracy, feed_dict={x : batch_xs, y :batch_ys}))

            setp += 1

















