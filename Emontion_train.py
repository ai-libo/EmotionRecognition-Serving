##################
#author:libo     #
#date :20180131  #
##################

#tensorflow version 1.0.1
#python version 3.5.2


import tensorflow as tf
import numpy as np
import cv2
import fnmatch,os,math
import re
from sklearn import cross_validation
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import os
import sys


learning_rate = 0.001
epochs = 100
#epochs = 1
batch_size = 1085
#batch_size = 10
#用来验证的样本数
#test_valid_size = 256
#test_valid_size = 36
n_classes = 8
dropout = 0.75
weights = {
    'wc1': tf.Variable(tf.random_normal([5 ,5, 1, 32]),dtype = tf.float32),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]),dtype = tf.float32),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128]),dtype = tf.float32),
    'wd1': tf.Variable(tf.random_normal([24 * 24 * 128 ,300]),dtype = tf.float32),
    'out': tf.Variable(tf.random_normal([300, n_classes]),dtype = tf.float32)
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32]),dtype = tf.float32),
    'bc2': tf.Variable(tf.random_normal([64]),dtype = tf.float32),
    'bc3': tf.Variable(tf.random_normal([128]),dtype = tf.float32),
    'bd1': tf.Variable(tf.random_normal([300]),dtype = tf.float32),
    'out': tf.Variable(tf.random_normal([n_classes]),dtype = tf.float32)
}
sess = tf.InteractiveSession()
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def conv_net(x, weights, biases, dropout):
    # 卷积层1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # 卷积层2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # 卷积层3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    # 全连接层 - 24 * 24 * 128  to 300
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # 输出分类 - 300 to 7
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'],name = 'out')
    return out
def strnum_convert(stra):
    convert_dict = {}
    labels = ["0","1","2","3","4","5","6","7"]
    num_label = list (range(0,8,1))
    convert_dict = dict(map(lambda x,y:[x,y], labels, num_label))
    return convert_dict[stra]
def numstr_convert(num):
    convert_dict = {}
    labels = ["0","1","2","3","4","5","6","7"]
    num_label = list (range(0,8,1))
    convert_dict = dict(map(lambda x,y:[x,y], num_label, labels))
    return convert_dict[num]

def one_hot(labels):
    labels_num = [strnum_convert(i) for i in labels ]
    batch_size = tf.size(labels_num)
    labels = tf.expand_dims(labels_num, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 8]), 1, 0)
    #all_hot_labels = tf.reshape(onehot_labels,(1,612))
    return onehot_labels

#def get_data ():
#d1 = datetime.now()
def total_create_data(url,batch_size):
    arrs1 = fnmatch.filter(os.listdir(url), '*.png')
    label_list = []
    feature_list = []
    for arr in arrs1:
        arr_re = re.findall((r'(\#)(.*)(\.)'),arr)
        label_list.append(one_hot(arr_re[0][1]).eval())
        img = cv2.imread(url + arr)
        #d2 = datetime.now()
        print (url+arr)
        #print "d2-d1=%s"%(d3-d2) 
        img_re = cv2.resize(img,(96,96),interpolation=cv2.INTER_CUBIC)
        pic = img_re.reshape([96,96,3])
        feature = np.reshape(np.average(pic,axis = 2),(96,96,1))
        feature_list.append(feature)
        #d3 = datetime.now()
        #print "d3-d2=%s"%(d3-d2) 
        #print (feature.shape)
    #print (label_list)
    total_x = np.array(feature_list)
    total_y = np.array(label_list).reshape(batch_size,8)
    return total_x,total_y


#url = '//data01//dataset//Emotion//liboEmontion-bak//train_batch1//'
url = 'train_test//'
total_x,total_y = total_create_data(url,batch_size)
train_x, test_x, train_y, test_y = cross_validation.train_test_split(total_x,total_y, test_size=0.2, random_state=0)

# tf Graph 输入
x = tf.placeholder(tf.float32, [None, 96, 96, 1],name='x')
y = tf.placeholder(tf.float32, [None, n_classes],name='y')
keep_prob = tf.placeholder(tf.float32)
# 模型logits
logits = conv_net(x, weights, biases, keep_prob)
# 损失和优化
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
predict_label = tf.argmax(logits, 1, name = "out2")
#correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
correct_pred = tf.equal(predict_label, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch = 1
#saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)


for epoch in range(epochs):
    sess.run(optimizer, feed_dict={
                x : train_x,
                y : train_y,
                keep_prob: dropout})

    logits_value = sess.run(logits, feed_dict={
        x : train_x,
        keep_prob: 1.})

    # 计算batch loss 和准确度 accuracy
    loss = sess.run(cost, feed_dict={
        x : train_x,
        y : train_y,
        keep_prob: 1.})

    valid_acc = sess.run(accuracy, feed_dict={
        x: test_x,
        y: test_y,
        keep_prob: 1.})
    print('Epoch {:>2}, Batch {:>3} -'
          'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
        epoch + 1,
        batch + 1,
        loss,
        valid_acc))

#测试准确度
test_acc = sess.run(accuracy,feed_dict = {
            x: test_x,
            y: test_y,
            keep_prob: 1.} )
print ('testing accuracy: {}'.format(test_acc))
#saver.save(sess, "model/Emontion_cnn.ckpt".format(test_acc))


# Export model
# WARNING(break-tutorial-inline-code): The following code snippet is
# in-lined in tutorials, please update tutorial documents accordingly
# whenever code changes.
export_path = 'model_save/export_model'
print ("####################################")
print ('Exporting trained model to'), export_path

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

classification_inputs  = tf.saved_model.utils.build_tensor_info(x)
classification_keepprob = tf.saved_model.utils.build_tensor_info(keep_prob)
classification_outputs = tf.saved_model.utils.build_tensor_info(predict_label)

classification_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            "classification_inputs":classification_inputs,
            "classification_keepprob":classification_keepprob
        },
        outputs={
            "classification_outputs":classification_outputs
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))


tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(predict_label)
tensor_info_k = tf.saved_model.utils.build_tensor_info(keep_prob)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tensor_info_x,
                 'keep_prob' : tensor_info_k},
        outputs={'output': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
         'predict_image':
          prediction_signature,
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

builder.save()
print("Build Done")

