import os
import pdb
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from data_process import DataInputTrain, DataInputTest
from model import Model
from sklearn.metrics import roc_auc_score
from data_prepare import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 64
test_batch_size = 64


answer_info_file="../dataset/info_answer.csv"
user_info_file="../dataset/info_user.csv"
ui_transaction_file="../dataset/zhihu1M.txt"
file_path=dataset_7_3_split(ui_transaction_file)
user_index,sex_index,province_index,city_index,answer_index,topic_index,author_index,user_count,sex_count,province_count,city_count,answer_count,topic_count,author_count=index_generate(file_path)
filePath=delete_answer_without_train_in_test(file_path,answer_index)
user_infos=user_infos_generate(user_index,sex_index,province_index,city_index,user_info_file)
answer_infos=answer_infos_generate(answer_index,topic_index,author_index,answer_info_file)
train_set,test_set=data_generate(filePath,user_index,answer_index,user_infos,answer_infos)
# train_set=[]
# test_set=[]
# user_count, item_count,sex_count,province_count,city_count,topic_count,author_count

def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d:d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    result = 1.0 - auc / (2.0 * tp2 * fp2)
    return result
def eval_offline_test(sess, model):
  y=[]
  y_pre=[]
  for _, uij in DataInputTest(test_set, test_batch_size):
    score_= model.eval_offline_test(sess, uij)
    for i in range(len(uij[0])):
        y_pre.append(score_[i])
        y.append(uij[11][i])
  Auc=roc_auc_score(y,y_pre)
  return Auc

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  model = Model(user_count, answer_count,sex_count, province_count, city_count, topic_count,author_count)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  auc_test=eval_offline_test(sess, model)
  print('test_auc: %.4f' % (auc_test))
  sys.stdout.flush()
  lr = 0.005
  start_time = time.time()
  for _ in range(5000):
    random.shuffle(train_set)
    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum_train = 0.0
    label_predict_positive=[]
    label_predict_negative=[]
    label_real_positive=[]
    label_real_negative=[]
    for _, uij in DataInputTrain(train_set, train_batch_size):
      positive_feature=uij[0]
      negative_feature=uij[1]
      predict_positive,loss_positive = model.train_offline(sess, positive_feature, lr,is_positive=True)
      predict_negative,loss_negative = model.train_offline(sess, negative_feature, lr,is_positive=False)
      loss=loss_positive*len(positive_feature)+loss_negative*len(negative_feature)
      loss_sum_train += loss
      if model.global_step.eval() % 336000 == 0:
        lr *= 0.9
      label_real_positive.append(positive_feature[11])
      label_real_negative.append(negative_feature[11])
      label_predict_positive.append(predict_positive)
      label_predict_negative.append(predict_negative)
    # auc for training process (epoch)
    auc_train = roc_auc_score([label_real_negative,label_real_positive], [label_predict_negative,label_predict_positive])
    auc_test = eval_offline_test(sess, model)
    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tAUC_train: %.4f\tAUC_test: %.4f' %
          (model.global_epoch_step.eval(), model.global_step.eval(),
           loss_sum_train / len(train_set), auc_train,auc_test))
    sys.stdout.flush()
    model.global_epoch_step_op.eval()


