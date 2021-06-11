import tensorflow as tf
import numpy as np

load_model_path = 'model/model_v1.ckpt'
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

def model_result(board):
    # input_x의 shape = (255,)의 벡터 형태   [자신이 둔 착수는 2, 타인이 둔 착수는 1 으로 표기된]
    input_X = board.flatten() / 2

    result = sess.run(logits, feed_dict={X: input_X[None, :]})
    result_mat = sigmoid(result).reshape([board_size, board_size])

    for row in range(4, 14):
        for col in range(15):
            # 내가 세로 4목이 있을 경우
            if ((board[row - 3][col] == 2.0 and board[row - 2][col] == 2.0 and board[row - 1][col] == 2.0 and board[row][col] == 2.0)) & \
                    ((board[row + 1][col] or board[row - 4][col]) == 0.0):
                result_mat[row + 1][col] = 0.99
            # 내가 세로 막힌 4목이 있을 경우
            elif ((board[row - 3][col] == 2.0 and board[row - 2][col] == 2.0 and board[row - 1][col] == 2.0 and board[row][col] == 2.0)) & \
                    ((board[row + 1][col] * board[row - 4][col]) == 0.0):
                if board[row + 1][col] == 0.0:
                    result_mat[row + 1][col] = 0.99
                else:
                    result_mat[row - 4][col] = 0.99
            # 내가 가로 4목이 있을 경우
            elif ((board[col][row - 3] == 2.0 and board[col][row - 2] == 2.0 and board[col][row - 1] == 2.0 and board[col][row] == 2.0)) & \
                    ((board[col][row + 1] or board[col][row - 4]) == 0.0):
                result_mat[col][row + 1] = 0.99
            # 내가 가로 막힌 4목이 있을 경우
            elif ((board[col][row - 3] == 2.0 and board[col][row - 2] == 2.0 and board[col][row - 1] == 2.0 and board[col][row] == 2.0)) & \
                    ((board[col][row + 1] * board[col][row - 4]) == 0.0):
                if board[col][row + 1] == 0.0:
                    result_mat[col][row + 1] = 0.99
                else:
                    result_mat[col][row - 4] = 0.99
        for col2 in range(4, 14):
            # 내가 대각 4목이 있을 경우
            if ((board[row - 3][col2 - 3] == 2.0 and board[row - 2][col2 - 2] == 2.0 and board[row - 1][col2 - 1] == 2.0 and board[row][col2] == 2.0)) & \
                    ((board[row + 1][col2 + 1] or board[row - 4][col2 - 4]) == 0.0):
                result_mat[row + 1][col2 + 1] = 0.99
            # 내가 대각 막힌 4목이 있을 경우
            elif (board[row - 3][col2 - 3] == 2.0 and board[row - 2][col2 - 2] == 2.0 and board[row - 1][col2 - 1] == 2.0 and board[row][col2] == 2.0) & \
                    ((board[row + 1][col2 + 1] * board[row - 4][col2 - 4]) == 0.0):
                if board[row + 1][col2 + 1] == 0.0:
                    result_mat[row + 1][col2 + 1] = 0.99
                else:
                    result_mat[row - 4][col2 - 4] = 0.99

    mat_flat = result_mat.flatten()
    sort_mat = mat_flat.argsort()

    top_10_idx = sort_mat[-10:]
    top_10_idx = np.flip(top_10_idx)

    top_10_perc = np.round(mat_flat[sort_mat][-10:]*100)
    top_10_perc = np.flip(top_10_perc)

    for i in range(len(top_10_perc)):
        if top_10_perc[i] < 30:
            top_10_perc[i] += 5
            break


    ten_idx = []
    for idx in top_10_idx:
        row = idx // 15
        col = idx % 15
        ten_idx.append((row, col))

    result = [str(ten_idx[i][0]) + '&' + str(ten_idx[i][1]) + '&' + str(int(top_10_perc[i])) for i in range(len(top_10_perc))]

    return result

board = np.array(
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 0
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 1
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 2
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 3
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 4
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 5
     [0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0.],  # 6
     [0., 0., 0., 0., 0., 0., 2., 1., 0., 0., 0., 0., 0., 0., 0.],  # 7
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 8
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 9
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 10
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 11
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 12
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 13
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]  # 14
)

"""    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 0
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 1
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 2
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 3
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 4
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 5
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 6
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 7
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 8
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 9
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 10
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 11
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 12
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 13
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]  # 14"""

result = model_result(board)

print(result)
