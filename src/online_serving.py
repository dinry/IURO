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
batch_size=32
user_index,sex_index,province_index,city_index,answer_index,topic_index,author_index,user_count,sex_count,province_count,city_count,answer_count,topic_count,author_count=index_generate(file_path)

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

save_model_dir="model_save"
class ImportRetention():
    """  Importing and running isolated TF graph """

    def __init__(self, save_model_dir,sv):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=4,
                                intra_op_parallelism_threads=4, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)
        with self.graph.as_default():
            saver=tf.train.import_meta_graph(save_model_dir + '/auc.meta', clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(save_model_dir))
            self.ui_predict = self.graph.get_tensor_by_name('ui_score/add:0')
            self.u = self.graph.get_tensor_by_name('init_param/u:0')
            self.sex = self.graph.get_tensor_by_name('init_param/sex:0')
            self.province = self.graph.get_tensor_by_name('init_param/province:0')
            self.city = self.graph.get_tensor_by_name('init_param/city:0')
            self.hist_click = self.graph.get_tensor_by_name('init_param/hist_click:0')
            self.hist_topic = self.graph.get_tensor_by_name('init_param/hist_topic:0')
            self.sl_hist_click = self.graph.get_tensor_by_name('init_param/sl_hist_click:0')
            self.target_item = self.graph.get_tensor_by_name('init_param/target_item:0')
            self.target_item_topic = self.graph.get_tensor_by_name('init_param/target_item_topic:0')
            self.short_term_item = self.graph.get_tensor_by_name('init_param/today_click:0')
            self.short_term_topic = self.graph.get_tensor_by_name('init_param/short_term_topic:0')
            self.short_term_sl = self.graph.get_tensor_by_name('init_param/short_term_sl:0')
    def run(self, _user,_gender,_province,_city, _hist_click,_hist_topic,_sl_click,_target_item,_target_item_topic,_short_term_item, _short_term_topic,_short_term_sl):
        return self.sess.run([self.ui_predict], feed_dict={self.u:_user,self.sex:_gender,self.province:_province,self.city:_city,self.hist_click:_hist_click,self.hist_topic:_hist_topic,self.sl_hist_click:_sl_click,self.target_item:_target_item,self.target_item_topic:_target_item_topic,self.short_term_item:_short_term_item, self.short_term_topic:_short_term_topic, self.short_term_sl:_short_term_sl})
class ImportCTR():
    """  Importing and running isolated TF graph """

    def __init__(self, save_model_dir,sv):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=4,
                                intra_op_parallelism_threads=4, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)
        with self.graph.as_default():
            saver=tf.train.import_meta_graph(save_model_dir + '/auc.meta', clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(save_model_dir))
            self.sigmoid=self.graph.get_tensor_by_name('item_embedding/Sigmoid:0')
            self.u = self.graph.get_tensor_by_name('init_param/u:0')
            self.sex = self.graph.get_tensor_by_name('init_param/sex:0')
            self.province = self.graph.get_tensor_by_name('init_param/province:0')
            self.city = self.graph.get_tensor_by_name('init_param/city:0')
            self.hist_click = self.graph.get_tensor_by_name('init_param/hist_click:0')
            self.hist_topic = self.graph.get_tensor_by_name('init_param/hist_topic:0')
            self.sl_hist_click = self.graph.get_tensor_by_name('init_param/sl_hist_click:0')
            self.target_item = self.graph.get_tensor_by_name('init_param/target_item:0')
            self.target_item_topic = self.graph.get_tensor_by_name('init_param/target_item_topic:0')
            self.short_term_item = self.graph.get_tensor_by_name('init_param/today_click:0')
            self.short_term_topic = self.graph.get_tensor_by_name('init_param/short_term_topic:0')
            self.short_term_sl = self.graph.get_tensor_by_name('init_param/short_term_sl:0')
    def run(self, _user,_gender,_province,_city, _hist_click,_hist_topic,_sl_click,_target_item,_target_item_topic,_short_term_item, _short_term_topic,_short_term_sl):
        return self.sess.run([self.sigmoid], feed_dict={self.u:_user,self.sex:_gender,self.province:_province,self.city:_city,self.hist_click:_hist_click,self.hist_topic:_hist_topic,self.sl_hist_click:_sl_click,self.target_item:_target_item,self.target_item_topic:_target_item_topic,self.short_term_item:_short_term_item, self.short_term_topic:_short_term_topic, self.short_term_sl:_short_term_sl})

def load_user(user,sex,province,city,hist_i,hist_t,hist_sl,short_term_i,short_term_t,short_term_sl):
    '''copy user data to match with candidate items'''
    user_feature=[]   # copy input
    return user_feature

# user data and item pool for online serving
answer_pool_file='answer_pool.data'
user_behavior_data="user_behavior.data"

# process target item
answer_pool_data=[]

# model path
save_retention_model_dir = 'user_retention_model_save'
retention_model_save_path = save_retention_model_dir + '/auc'
save_ctr_model_dir = 'ctr_model_save'
ctr_model_save_path = save_ctr_model_dir + '/auc'

# write file
ui_retention_file="ui_retention_for_online_serving"
ctr_file="ctr_for_online_serving"

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
is_chief = (FLAGS.task_index == 0)
cluster = tf.train.ClusterSpec({"ps": FLAGS.ps_hosts.split(","), "worker": FLAGS.worker_hosts.split(",")})
server = tf.train.Server(cluster.as_cluster_def(),
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)
model_ui_retention= ImportRetention(save_retention_model_dir,server.target)
model2 = ImportCTR(save_ctr_model_dir,server.target)

for line in user_behavior_data:
    flds = line.strip().split('\t')
    user=flds[0]
    sex=flds[1]
    province=flds[2]
    city=flds[3]
    hist_i=flds[4]
    hist_t=flds[5]
    hist_sl=flds[6]
    short_term_i=flds[7]
    short_term_t=flds[8]
    short_term_sl=flds[9]
    u_feature = load_user(user,sex,province,city,hist_i,hist_t,hist_sl,short_term_i,short_term_t,short_term_sl)
    result_retention = model_ui_retention.run(u_feature[0],u_feature[1],u_feature[2],u_feature[3],u_feature[4],u_feature[5],u_feature[6],answer_pool_data[0],answer_pool_data[1])
    index=result_retention[0].argsort()[::-1]
    ui_retention_top_500=np.array(answer_pool_data)[index[0:500]]
    print(ui_retention_top_500)


# join with ctr model

# for file_u in user_behavior_data:
#     ui_write=open(ui_retention_file_path+"/"+file_u,'w')
#     for line in open(user_behavior_file + "/" + file_u):
#         flds = line.strip().split('\t')
#         u_feature = load_user(flds[0], flds[1], flds[2], flds[3], flds[4], flds[5])
#         result_retention = model1.run(u_feature[0],u_feature[1],u_feature[2],u_feature[3],u_feature[4],u_feature[5],u_feature[6],doc_feature[0],doc_feature[1])
#         result_ctr = model2.run(u_feature[0], u_feature[1], u_feature[2], u_feature[3], u_feature[4], u_feature[5],
#                                      u_feature[6], doc_feature[0], doc_feature[1])
#     index=result_ctr[0].argsort()[::-1]
#     score_name=np.array(doc_list)[index[0:500]]
#     score_merge=0.2*3*result_ctr[0][index[0:500]]+0.8*result_retention[0][index[0:500]]
#     score_retention=dict(zip(score_name,score_merge))
#     score_retention_rank=sorted(score_retention.items(), key=lambda x: x[1], reverse=True)
#     score_retention_rank=score_retention_rank[0:30]
#     print("part-end",file=sys.stderr)
#     ui_write.flush()
#     ui_write.close()

