from __future__ import print_function
import tensorflow as tf
import math
from tensorflow.contrib import layers
import numpy as np
import random
from model import Model
from data_process import *
from data_prepare import *
import pickle
import datetime
import sys
import os
answer_info_file="../dataset/answer_infos.txt"
user_info_file="../dataset/user_infos.txt"
ui_transaction_file="../dataset/zhihu1M.txt"
user_index, sex_index, province_index, city_index, answer_index, topic_index, author_index, user_count, sex_count, province_count, city_count, answer_count, topic_count, author_count = index_generate(
    ui_transaction_file, answer_info_file, user_info_file)


def generate_index(user_index, sex_index, province_index, city_index, answer_index, topic_index):
    '''
    generate index for recommend efficiently, including index->user, index->item,
    '''
    user_index2id={}
    user_id2index={}
    for user in user_index.keys():
        user_index2id[user]=user_index[user]
        user_id2index[user_index[user]]=user

    sex_index2id={}
    sex_id2index={}
    for sex in sex_index.keys():
        sex_index2id[sex]=sex_index[sex]
        sex_id2index[sex_index[sex]]=sex

    province_index2id={}
    province_id2index={}
    for province in province_index.keys():
        province_index2id[province]=province_index[province]
        province_id2index[province_index[province]]=province

    city_index2id={}
    city_id2index={}
    for city in city_index.keys():
        city_index2id[city]=city_index[city]
        city_id2index[city_index[city]]=city

    answer_index2id={}
    answer_id2index={}
    for answer in answer_index.keys():
        answer_index2id[answer]=answer_index[answer]
        answer_id2index[answer_index[answer]]=answer

    topic_index2id={}
    topic_id2index={}
    for topic in topic_index.keys():
        topic_index2id[topic]=topic_index[topic]
        topic_id2index[topic_index[topic]]=topic
    return user_index2id, user_id2index,sex_index2id,sex_id2index,province_index2id,province_id2index,city_index2id,city_id2index,answer_index2id,answer_id2index,topic_index2id,topic_id2index

class ImportRetention():
    """  Importing and running isolated TF graph """

    def __init__(self, save_model_dir):
        self.graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=4,
                                intra_op_parallelism_threads=4, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph,config=config)
        with self.graph.as_default():
            saver=tf.compat.v1.train.import_meta_graph(save_model_dir + '/auc.meta', clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(save_model_dir))
            self.ui_predict = self.graph.get_tensor_by_name('ui_score/add:0')
            self.u = self.graph.get_tensor_by_name('init_param/user:0')
            self.sex = self.graph.get_tensor_by_name('init_param/sex:0')
            self.province = self.graph.get_tensor_by_name('init_param/province:0')
            self.city = self.graph.get_tensor_by_name('init_param/city:0')
            self.hist_click = self.graph.get_tensor_by_name('init_param/hist_click:0')
            self.hist_topic = self.graph.get_tensor_by_name('init_param/hist_topic:0')
            self.sl_hist_click = self.graph.get_tensor_by_name('init_param/sl_hist_click:0')
            self.target_item = self.graph.get_tensor_by_name('init_param/target_item:0')
            self.target_item_topic = self.graph.get_tensor_by_name('init_param/target_item_topic:0')
    def run(self, _user,_gender,_province,_city, _hist_click,_hist_topic,_sl_click,_target_item,_target_item_topic):
        return self.sess.run([self.ui_predict], feed_dict={self.u:_user,self.sex:_gender,self.province:_province,self.city:_city,self.hist_click:_hist_click,self.hist_topic:_hist_topic,self.sl_hist_click:_sl_click,self.target_item:_target_item,self.target_item_topic:_target_item_topic})


# user data and item pool for online serving
candidate_item_pool_file='../dataset/online_serving/candidate_item_pool.data'
user_behavior = pickle.load(open('../dataset/online_serving/user_behavior.data', 'rb'))

# model path
save_retention_model_dir = '../model_save'
retention_model_save_path = save_retention_model_dir + '/auc'
#generate index
user_index2id, user_id2index,sex_index2id,sex_id2index,province_index2id,province_id2index,city_index2id,city_id2index,answer_index2id,answer_id2index,topic_index2id,topic_id2index=generate_index(user_index, sex_index, province_index, city_index, answer_index, topic_index)

model_ui_retention= ImportRetention(save_retention_model_dir)
def load_candidate_item(candidate_item_pool_file,answer_index2id,topic_index2id):
    '''
    input: candidate item pool
    output: candidate_items_id, candidate_items_topic
    '''
    candidate_items_id=[]
    candidate_items_topic=[]
    for line in open(candidate_item_pool_file,'r'):
        line = line[:-1].strip('\n').split('\t')
        candidate_items_id.append(answer_index2id[line[0]])
        candidate_items_topic.append(topic_index2id[line[1]])
    return candidate_items_id, candidate_items_topic

result_ui_retention_score = open("../dataset/result_ui_score_for_online_serving","w")
for user_behav in user_behavior:
    # load candidate pool for dynamic items
    candidate_items_id, candidate_items_topic=load_candidate_item(candidate_item_pool_file,answer_index2id,topic_index2id)
    uu=[]
    for i in range(len(candidate_items_id)):
        uu.append(user_behav)
    ui=UserProfile_OnlineServing(uu)
    ui_retention_score = model_ui_retention.run(ui[0],ui[1],ui[2],ui[3],ui[4],ui[5],ui[6], candidate_items_id,candidate_items_topic)

    # rank ui retention score
    index=ui_retention_score[0].argsort()[::-1]
    # choose the best 50 items to serve online.
    ui_retention_score_top_50=np.array(ui_retention_score[0])[index[0:50]]
    candidate_items_id_top_50=np.array(candidate_items_id)[index[0:50]]
    for i in range(50):
        result_ui_retention_score.write(str(user_id2index[user_behav[0]]) + "\t" + str(answer_id2index[candidate_items_id_top_50[i]]) + "\t" + str(ui_retention_score_top_50[i]) + "\n")
        result_ui_retention_score.flush()
        # user, item, ui_retention_score






