import tensorflow as tf
import random
import os
import numpy as np
import pandas as pd

def data_aug(board_data): # 1개의 데이터를 회전, 반전으로 12개로 증강
    board_stack = []

    board_mat = board_data.reshape([board_size,board_size])
    
    # rotate
    board_stack.append(board_mat)
    board_stack.append(np.rot90(board_mat))
    board_stack.append(np.rot90(board_mat,2))
    board_stack.append(np.rot90(board_mat,3))

    board_fliplr = np.fliplr(board_mat)
    board_flipud = np.flipud(board_mat)
    
    # rotate and filp
    board_stack.append(board_fliplr)
    board_stack.append(np.rot90(board_fliplr))
    board_stack.append(np.rot90(board_fliplr,2))
    board_stack.append(np.rot90(board_fliplr,3))

    board_stack.append(board_flipud)
    board_stack.append(np.rot90(board_flipud))
    board_stack.append(np.rot90(board_flipud,2))
    board_stack.append(np.rot90(board_flipud,3))

    return board_stack


### set parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
board_size = 15
file_path = 'training_data/txt/' # 학습 기보 저장된 폴더
save_file = 'model/model_white_v2.ckpt' # 모델 저장 폴더 / 파일명

### 학습 데이터 저장 공간
board_x_stack = []
board_y_stack = []

keep_prob = tf.placeholder(tf.float32)


### set network
X = tf.placeholder(tf.float32, [None, board_size*board_size])
X_img = tf.reshape(X, [-1, board_size, board_size, 1])
Y = tf.placeholder(tf.float32, [None, board_size*board_size])
 
W1 = tf.Variable(tf.random_normal([7,7,1,64], stddev=0.1))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME') 
L1 = tf.nn.relu(L1)
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
 
W2 = tf.Variable(tf.random_normal([5,5,64,32], stddev = 0.1))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
#L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
 
W3 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L4 = tf.nn.conv2d(L3, W4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.relu(L4)
#L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


W5 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L5 = tf.nn.conv2d(L4, W5, strides=[1,1,1,1], padding='SAME')
L5 = tf.nn.relu(L5)
#L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L6 = tf.nn.conv2d(L5, W6, strides=[1,1,1,1], padding='SAME')
L6 = tf.nn.relu(L6)

W7 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L7 = tf.nn.conv2d(L6, W7, strides=[1,1,1,1], padding='SAME')
L7 = tf.nn.relu(L7)

W8 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L8 = tf.nn.conv2d(L7, W8, strides=[1,1,1,1], padding='SAME')
L8 = tf.nn.relu(L8)

W9 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
L9 = tf.nn.conv2d(L8, W9, strides=[1,1,1,1], padding='SAME')
L9 = tf.nn.relu(L9)
#L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

L9 = tf.reshape(L9, [-1, board_size * board_size * 32])



W10 = tf.get_variable("W10", shape=[board_size * board_size * 32, board_size*board_size],
                      initializer = tf.contrib.layers.xavier_initializer())

b10 = tf.Variable(tf.random_normal([board_size * board_size]))
logits = tf.matmul(L9,W10) + b10
 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 

# data load / augmentation
df = pd.read_csv('tot_data_plus_result.csv')

for i in range(len(df)): # 일단 10000개로 학습 바꿀 것 len(df)로
    # read data
    board_data = np.array(df.iloc[i][1:-1])
    result = df.iloc[i][-1]

    if i % 500 == 0:
        print(i)

    # data augmentation
    # 흑이 이긴 경우만 train한다
    ##############  result   0:백승 1:흑승  ##############
    if result != 0:
        continue

    else:
        board_stack = data_aug(board_data)

        for board_i in range(len(board_stack)):
            board_vec = board_stack[board_i].flatten()
            max_step = np.max(board_vec)

            for data_cnt in range(1,int(round(max_step))):
                board_x = np.zeros([board_size*board_size],dtype='f')
                board_y = np.zeros([board_size*board_size],dtype='f')

                for i in range(len(board_vec)):

                    ############# 흑/백 학습 데이터 생성
                    # 학습시에는 어느턴이든 항상 자기 돌은 1 상대돌은 0.5 으로 표현하게 한다
                    if data_cnt%2 != 0:
                        if board_vec[i] > 0 and board_vec[i] <= data_cnt: # 배경 skip
                            if board_vec[i]%2 != 0:
                                # 흑
                                board_x[i] = 1
                            else:
                                board_x[i] = 2

                        if board_vec[i] == data_cnt + 1:
                            board_y[i] = 1

                    else:
                        if board_vec[i] > 0 and board_vec[i] <= data_cnt: # 배경 skip
                            if board_vec[i]%2 == 0:
                                # 백
                                board_x[i] = 1
                            else:
                                board_x[i] = 2

                        if board_vec[i] == data_cnt + 1:
                            board_y[i] = 1

                if (np.sum(board_y)) != 1:
                    print('학습데이터에 문제가 있습니다.')

                board_x_stack.append(board_x/2)
                board_y_stack.append(board_y)


total_batch = (round(len(board_x_stack)/100)) - 1
print('학습데이터 생성이 완료 되었습니다: ', total_batch)

######### 학습 시작
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
 
print("Learning started, It takes sometime.")
for epoch in range(training_epochs):
    avg_cost = 0
    
    batch_xs = np.zeros([batch_size,board_size*board_size],dtype='f')
    batch_ys = np.zeros([batch_size,board_size*board_size],dtype='f')

    for i in range(total_batch):
        
        for j in range(batch_size):

            batch_xs[j,:] = board_x_stack[j+i*batch_size]
            batch_ys[j,:] = board_y_stack[j+i*batch_size]
        
        
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
 
    print('Epoch;', '%04d' % (epoch +1), 'cost =', '{:.9f}'.format(avg_cost))
 
print('학습이 완료 되었습니다.')



######### 학습 모델 저장
saver = tf.train.Saver()
saver.save(sess, save_file)


print('모델저장이 완료 되었습니다.')