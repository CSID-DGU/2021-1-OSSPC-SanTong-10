import tensorflow as tf
import numpy as np

load_model_path = 'model/model_v2.ckpt'
board_size = 15

X = tf.placeholder(tf.float32, [None, board_size*board_size])
X_img = tf.reshape(X, [-1, board_size, board_size, 1])

W1 = tf.Variable(tf.random_normal([7, 7, 1, 64], stddev=0.1))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=0.1))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
# L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
# L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
# L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


W5 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=keep_prob)


W6 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(L6)

W7 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
L7 = tf.nn.relu(L7)

W8 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME')
L8 = tf.nn.relu(L8)

W9 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
L9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME')
L9 = tf.nn.relu(L9)
# L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

L9 = tf.reshape(L9, [-1, board_size * board_size * 32])

W10 = tf.get_variable("W10", shape=[board_size * board_size * 32, board_size * board_size],
                      initializer=tf.contrib.layers.xavier_initializer())

b10 = tf.Variable(tf.random_normal([board_size * board_size]))
logits = tf.matmul(L9, W10) + b10

saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, load_model_path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def model_result(input_X):
    # input_x의 shape = (255,)의 벡터 형태   [자신이 둔 착수는 0.5, 타인이 둔 착수는 1.0 으로 표기된]
    result = sess.run(logits, feed_dict={X: input_X[None, :]})
    result_mat = sigmoid(result).reshape([board_size, board_size])
    return result_mat

input_X = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0.5,1.0,0,0,0,0,0,0,
           0,0,0,0,0,0,0.5,1.0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

result_mat = model_result(input_X)
print('result_mat:', np.round(result_mat, 2))