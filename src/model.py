import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)
import pdb
class Model(object):

    def __init__(self, user_count,sex_count,province_count, city_count, item_count, topic_count):

        with tf.name_scope('init_param'):
            self.regularizer=tf.contrib.layers.l2_regularizer(0.0001)
            self.hidden_units=64
            self.aug_weight=0.3
            self.neg_weight=0.3
            self.margin=2.5
            self.lambad_1=1
            self.lambad_2=1
            self.ui_prop=0.5

            self.global_step=tf.Variable(0.0, trainable=False)

            self.user_count=user_count
            self.sex_count=sex_count
            self.province_count=province_count
            self.city_count=city_count

            self.item_count=item_count
            self.topic_count=topic_count

            # placeholder
            self.lr = tf.placeholder(tf.float64, [])
            self.user = tf.placeholder(tf.int32, [None, ],name="user")  # [B]
            self.sex=tf.placeholder(tf.int32, [None, ],name="sex")
            self.province = tf.placeholder(tf.int32, [None, ],name="province")
            self.city = tf.placeholder(tf.int32, [None, ],name="city")

            self.hist_click = tf.placeholder(tf.int32, [None, None],name="hist_click")  # [B, T]
            self.hist_topic = tf.placeholder(tf.int32, [None, None],name="hist_topic")  # [B, T]
            self.sl_hist_click=tf.placeholder(tf.int32, [None, ],name="sl_hist_click")  # [B]

            self.rational_click = tf.placeholder(tf.int32, [None, None],name="rational_click")  # [B, T]
            self.rational_topic = tf.placeholder(tf.int32, [None, None],name="rational_topic")  # [B, T]
            self.rational_sl=tf.placeholder(tf.int32, [None, ],name="rational_sl")  # [B]

            self.short_term_click = tf.placeholder(tf.int32, [None, None],name="short_term_click")  # [B, T]
            self.short_term_topic = tf.placeholder(tf.int32, [None, None],name="short_term_topic")  # [B, T]
            self.short_term_sl=tf.placeholder(tf.int32, [None, ],name="short_term_sl")  # [B]

            self.long_term_click = tf.placeholder(tf.int32, [None, None],name="long_term_click")  # [B, T]
            self.long_term_topic = tf.placeholder(tf.int32, [None, None],name="long_term_topic")  # [B, T]
            self.long_term_sl=tf.placeholder(tf.int32, [None, ],name="long_term_sl")  # [B]

            self.y = tf.placeholder(tf.float32, [None, ],name="y")  # [B]
            # {0,1}: y_sigmoid=1 denotes retention, otherwise not.
            self.y_sigmoid=tf.placeholder(tf.float32, [None, ],name="y_sigmoid")  # [B]
            self.supervised_signals_click = tf.placeholder(tf.float32, [None, ],name="supervised_signals_click")  # [B] ss_click
            self.supervised_signals_impression = tf.placeholder(tf.float32, [None, ],name="supervised_signals_impression")  # [B] #ss_impression

            # predict for online serving (used in item candidates)
            self.target_item=tf.placeholder(tf.int32, [None, ],name="target_item")  # [B]
            self.target_item_topic = tf.placeholder(tf.int32, [None, ], name="target_item_topic")  # [B]

            # clicks in next three days for training
            self.future_click=tf.placeholder(tf.int32, [None, None],name="future")  # [B, T]
            self.future_topic = tf.placeholder(tf.int32, [None, None],name="future_topic")  # [B, T]
            self.sl_future_click=tf.placeholder(tf.int32, [None, ],name="sl_future_click")  # [B]

        with tf.name_scope('embedding_init'):
            self.user_emb_w = tf.get_variable("user_emb_w", [self.user_count, self.hidden_units // 2],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.item_emb_w = tf.get_variable("item_emb_w", [self.item_count, self.hidden_units // 2],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.sex_emb_w=tf.get_variable("sex_emb_w",[self.sex_count,16],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.province_emb_w=tf.get_variable("province_emb_w",[self.province_count,16],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.city_emb_w = tf.get_variable("city_emb_w", [self.city_count, 32],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.topic_emb_w = tf.get_variable("topic_emb_w", [self.topic_count, 32],initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            self.item_b = tf.get_variable("item_b", [self.item_count],initializer=tf.constant_initializer(0.0))

        with tf.name_scope('user_embedding'):
            user_id_embedding=tf.nn.embedding_lookup(self.user_emb_w, self.user)
            user_sex_emb=tf.nn.embedding_lookup(self.sex_emb_w,self.sex)
            user_province_emb=tf.nn.embedding_lookup(self.province_emb_w,self.province)
            user_city_emb=tf.nn.embedding_lookup(self.city_emb_w,self.city)
            self.user_profile=tf.concat([user_id_embedding,user_sex_emb,user_province_emb,user_city_emb],axis=1)

        with tf.name_scope('ui_score'):
            target_average = tf.TensorArray(
                dtype=tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False
            )
            # ui retention modeling, ui_retention_score generated by ui_scorer, which can be used when online serving.
            def ui_scorer(user_profile, target_item, target_item_topic, hist_item, hist_topicg, sl):
                target_item_emb = tf.nn.embedding_lookup(self.item_emb_w, target_item)
                target_item_topic_emb = tf.nn.embedding_lookup(self.topic_emb_w, target_item_topic)
                target_item_bias = tf.gather(self.item_b, target_item)
                hist = tf.nn.embedding_lookup(self.item_emb_w, hist_item)
                mask = tf.sequence_mask(sl, tf.shape(hist)[1], dtype=tf.float32)  # [B, T]
                mask = tf.expand_dims(mask, -1)  # [B, T, 1]
                hist *= mask
                hist = tf.reduce_sum(hist, 1)
                hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(sl, 1), [1, 32]), tf.float32))
                hist = tf.layers.batch_normalization(inputs=hist, name='hist_1', reuse=tf.AUTO_REUSE)
                hist = tf.reshape(hist, [-1, self.hidden_units // 2])
                hist = tf.layers.dense(hist, 128, name='hist_2', reuse=tf.AUTO_REUSE)
                hist_a = tf.nn.embedding_lookup(self.topic_emb_w, hist_topicg)
                hist_a *= mask
                hist_a = tf.reduce_sum(hist_a, 1)
                hist_a = tf.div(hist_a, tf.cast(tf.tile(tf.expand_dims(sl, 1), [1, 32]), tf.float32))
                hist_a = tf.layers.batch_normalization(inputs=hist_a, name='hist_topic_1', reuse=tf.AUTO_REUSE)
                hist_a = tf.reshape(hist_a, [-1, self.hidden_units // 2])
                hist_a = tf.layers.dense(hist_a, 128, name='hist_topic_2', reuse=tf.AUTO_REUSE)
                field = tf.concat([user_profile, hist, hist_a, target_item_emb, target_item_topic_emb], axis=1)
                din_i = tf.layers.batch_normalization(inputs=field, name='ui_b1', reuse=tf.AUTO_REUSE)
                d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='ui_b2', reuse=tf.AUTO_REUSE)
                d_layer_3_i = tf.layers.dense(d_layer_1_i, 1, activation=None, name='ui_b3', reuse=tf.AUTO_REUSE)
                d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
                logits = target_item_bias + d_layer_3_i
                return logits

            init_loop_var = (
                self.user_profile, self.hist_click, self.hist_topic, 0, target_average, self.short_term_click,
                self.short_term_topic)

            def continue_loop_condition(user_profile, hist_behavior, hist_topic, step, target_average, short_term_click,
                                        short_term_topic):
                return tf.less(step, tf.shape(short_term_click)[1])

            def loop_body(user_profile, hist_behavior, hist_topic, step, target_average, short_term_click, short_term_topic):
                target_item = short_term_click[:, step]
                target_item_topic = short_term_topic[:, step]
                logits = ui_scorer(user_profile, target_item, target_item_topic, hist_behavior, hist_topic, self.sl_hist_click)
                target_average = target_average.write(step, logits)
                return user_profile, hist_behavior, hist_topic, step + 1, target_average, short_term_click, short_term_topic

            user_profile_v1, hist_behavior_v1, hist_topic_v1, step_v1, target_average_v1, short_term_click_v1, short_term_topic_v1 = tf.while_loop(
                cond=continue_loop_condition, body=loop_body, loop_vars=init_loop_var)
            self.logit = target_average_v1.stack()
            self.score_ui = tf.transpose(self.logit)
            self.ui_predict = ui_scorer(self.user_profile, self.target_item, self.target_item_topic, self.hist_click,
                                       self.hist_topic, self.sl_hist_click)


        with tf.name_scope('attention'):
            def attention_w_similarity(queries, keys, keys_length, logit):
                queries_hidden_units = queries.get_shape().as_list()[-1]
                queries = tf.tile(queries, [1, tf.shape(keys)[1]])
                queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
                outputs = tf.reduce_sum(queries * keys, axis=2, keep_dims=True)
                outputs = tf.reshape(outputs, [tf.shape(queries)[0], tf.shape(keys)[1]])
                similarity = tf.reshape(outputs, [tf.shape(queries)[0], tf.shape(keys)[1]])
                key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
                paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
                outputs = tf.where(key_masks, outputs, paddings)
                outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
                outputs = tf.nn.softmax(outputs)
                attention = outputs
                index = tf.argmax(outputs, axis=1)
                outputs = tf.reduce_sum(tf.multiply(outputs, logit), axis=1)
                return outputs, attention, similarity, index
            def attention_generate(target_item, target_item_topic):
                target_item_emb = tf.nn.embedding_lookup(self.item_emb_w, target_item)
                target_topic_emb = tf.nn.embedding_lookup(self.topic_emb_w, target_item_topic)
                keys = tf.concat([target_item_emb, target_topic_emb], axis=2)
                keys = tf.layers.dense(keys, self.hidden_units, activation=tf.nn.relu, use_bias=True, name='key_0',
                                       reuse=tf.AUTO_REUSE)
                keys = tf.layers.dense(keys, self.hidden_units // 2, activation=tf.nn.sigmoid, use_bias=True,
                                       name='key_1', reuse=tf.AUTO_REUSE)
                h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist_click)
                h_topic_emb = tf.nn.embedding_lookup(self.topic_emb_w, self.hist_topic)
                mask = tf.sequence_mask(self.sl_hist_click, tf.shape(h_emb)[1], dtype=tf.float32)  # [B, T]
                mask = tf.expand_dims(mask, -1)  # [B, T, 1]
                mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B, T, H]
                h_emb *= mask  # [B, T, H]
                hist = tf.reduce_sum(h_emb, 1)
                hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl_hist_click, 1), [1, 32]), tf.float32))
                hist = tf.layers.batch_normalization(inputs=hist, name='hist_1', reuse=tf.AUTO_REUSE)
                hist = tf.reshape(hist, [-1, self.hidden_units // 2])
                hist = tf.layers.dense(hist, 128, name='hist_2', reuse=tf.AUTO_REUSE)
                h_topic_emb *= mask  # [B, T, H]
                hist_a = tf.reduce_sum(h_topic_emb, 1)
                hist_a = tf.div(hist_a, tf.cast(tf.tile(tf.expand_dims(self.sl_hist_click, 1), [1, 32]), tf.float32))
                hist_a = tf.layers.batch_normalization(inputs=hist_a, name='hist_topic_1', reuse=tf.AUTO_REUSE)
                hist_a = tf.reshape(hist_a, [-1, self.hidden_units // 2])
                hist_a = tf.layers.dense(hist_a, 128, name='hist_topic_2', reuse=tf.AUTO_REUSE)
                query = tf.concat([self.user_profile, hist, hist_a], axis=1)
                query = tf.layers.dense(query, self.hidden_units, activation=tf.nn.relu, use_bias=True, name='query_0',
                                        reuse=tf.AUTO_REUSE)
                query = tf.layers.dense(query, self.hidden_units // 2, activation=tf.nn.sigmoid, use_bias=True,
                                        name='query_1', reuse=tf.AUTO_REUSE)
                return query, keys

            query, keys = attention_generate(self.short_term_click, self.short_term_topic)
            self.keys = keys
            self.logits_click, self.att_hist, self.similarity, self.index = attention_w_similarity(query, keys,self.short_term_sl,self.score_ui)

        with tf.name_scope('rational_positive'):
            future_emb = tf.nn.embedding_lookup(self.item_emb_w, self.future_click)
            future_topic_emb = tf.nn.embedding_lookup(self.topic_emb_w, self.future_topic)
            mask = tf.sequence_mask(self.sl_future_click, tf.shape(future_emb)[1], dtype=tf.float32)  # [B, T]
            mask = tf.expand_dims(mask, -1)  # [B, T, 1]
            mask = tf.tile(mask, [1, 1, tf.shape(future_emb)[2]])  # [B, T, H]
            future_emb *= mask  # [B, T, H]
            future = tf.reduce_sum(future_emb, 1)
            future = tf.div(future, tf.cast(tf.tile(tf.expand_dims(self.sl_future_click, 1), [1, 32]), tf.float32))
            future = tf.layers.batch_normalization(inputs=future, name='f_1', reuse=tf.AUTO_REUSE)
            future = tf.reshape(future, [-1, self.hidden_units // 2])
            future = tf.layers.dense(future, self.hidden_units // 2, name='f_2', reuse=tf.AUTO_REUSE)
            future_topic_emb *= mask  # [B, T, H]
            future_a = tf.reduce_sum(future_topic_emb, 1)
            future_a = tf.div(future_a, tf.cast(tf.tile(tf.expand_dims(self.sl_future_click, 1), [1, 32]), tf.float32))
            future_a = tf.layers.batch_normalization(inputs=future_a, name='f_3', reuse=tf.AUTO_REUSE)
            future_a = tf.reshape(future_a, [-1, self.hidden_units // 2])
            future_a = tf.layers.dense(future_a, self.hidden_units // 2, name='f_4', reuse=tf.AUTO_REUSE)
            query_future = tf.concat([future, future_a], axis=1)
            query_future = tf.layers.dense(query_future, self.hidden_units, activation=tf.nn.relu, use_bias=True,
                                           name='f_5', reuse=tf.AUTO_REUSE)
            query_future = tf.layers.dense(query_future, self.hidden_units // 2, activation=tf.nn.sigmoid,
                                           use_bias=True, name='f_6', reuse=tf.AUTO_REUSE)
            item = tf.nn.embedding_lookup(self.item_emb_w, self.rational_click)
            topic = tf.nn.embedding_lookup(self.topic_emb_w, self.rational_topic)
            keys_b = tf.concat([item, topic], axis=2)
            keys_b = tf.layers.dense(keys_b, self.hidden_units, activation=tf.nn.relu, use_bias=True, name='f_7',
                                     reuse=tf.AUTO_REUSE)
            keys_b = tf.layers.dense(keys_b, self.hidden_units // 2, activation=tf.nn.sigmoid, use_bias=True,
                                     name='f_8', reuse=tf.AUTO_REUSE)

            def similarity_future(queries, keys, keys_length):
                queries_hidden_units = queries.get_shape().as_list()[-1]
                queries = tf.tile(queries, [1, tf.shape(keys)[1]])
                queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
                outputs = tf.reduce_sum(queries * keys, axis=2, keep_dims=True)
                outputs = tf.reshape(outputs, [tf.shape(queries)[0], tf.shape(keys)[1]])
                key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
                paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
                outputs = tf.where(key_masks, outputs, paddings)
                outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
                outputs = tf.nn.softmax(outputs)
                att = outputs

                def less_5_b():
                    return self.rational_click, self.rational_topic

                def greater_5_b(att):
                    topk = tf.nn.top_k(att, 5)
                    line = tf.cast(tf.reshape(tf.range(tf.shape(self.rational_click)[0]), [-1, 1]), dtype=tf.int32)
                    indices_0 = tf.concat([line, tf.reshape(topk.indices[:, 0], [-1, 1])], axis=1)
                    click_indices_0 = tf.reshape(tf.gather_nd(self.rational_click, indices=indices_0),
                                                 [tf.shape(self.rational_click)[0], -1])
                    cate_indices_0 = tf.reshape(tf.gather_nd(self.rational_topic, indices=indices_0),
                                                [tf.shape(self.rational_click)[0], -1])

                    indices_1 = tf.concat([line, tf.reshape(topk.indices[:, 1], [-1, 1])], axis=1)
                    click_indices_1 = tf.reshape(tf.gather_nd(self.rational_click, indices=indices_1),
                                                 [tf.shape(self.rational_click)[0], -1])
                    cate_indices_1 = tf.reshape(tf.gather_nd(self.rational_topic, indices=indices_1),
                                                [tf.shape(self.rational_click)[0], -1])

                    indices_2 = tf.concat([line, tf.reshape(topk.indices[:, 2], [-1, 1])], axis=1)
                    click_indices_2 = tf.reshape(tf.gather_nd(self.rational_click, indices=indices_2),
                                                 [tf.shape(self.rational_click)[0], -1])
                    cate_indices_2 = tf.reshape(tf.gather_nd(self.rational_topic, indices=indices_2),
                                                [tf.shape(self.rational_click)[0], -1])

                    indices_3 = tf.concat([line, tf.reshape(topk.indices[:, 3], [-1, 1])], axis=1)
                    click_indices_3 = tf.reshape(tf.gather_nd(self.rational_click, indices=indices_3),
                                                 [tf.shape(self.rational_click)[0], -1])
                    cate_indices_3 = tf.reshape(tf.gather_nd(self.rational_topic, indices=indices_3),
                                                [tf.shape(self.rational_click)[0], -1])

                    indices_4 = tf.concat([line, tf.reshape(topk.indices[:, 4], [-1, 1])], axis=1)
                    click_indices_4 = tf.reshape(tf.gather_nd(self.rational_click, indices=indices_4),
                                                 [tf.shape(self.rational_click)[0], -1])
                    cate_indices_4 = tf.reshape(tf.gather_nd(self.rational_topic, indices=indices_4),
                                                [tf.shape(self.rational_click)[0], -1])
                    click_indices = tf.concat(
                        [click_indices_0, click_indices_1, click_indices_2, click_indices_3, click_indices_4], axis=1)
                    cate_indices = tf.concat(
                        [cate_indices_0, cate_indices_1, cate_indices_2, cate_indices_3, cate_indices_4], axis=1)
                    return click_indices, cate_indices
                rational_click_top5, rational_topic_top5 = tf.cond(tf.less(tf.reduce_max(keys_length), 5),
                                                                  lambda: less_5_b(), lambda: greater_5_b(att))
                return rational_click_top5, rational_topic_top5

            self.rational_click_top5, self.rational_topic_top5 = similarity_future(query_future, keys_b,
                                                                                  self.rational_sl)
            logit_0 = ui_scorer(self.user_profile, self.rational_click_top5[:, 0], self.rational_topic_top5[:, 0],
                               self.hist_click, self.hist_topic, self.sl_hist_click)
            logit_1 = ui_scorer(self.user_profile, self.rational_click_top5[:, 1], self.rational_topic_top5[:, 1],
                               self.hist_click, self.hist_topic, self.sl_hist_click)
            logit_2 = ui_scorer(self.user_profile, self.rational_click_top5[:, 2], self.rational_topic_top5[:, 2],
                               self.hist_click, self.hist_topic, self.sl_hist_click)
            logit_3 = ui_scorer(self.user_profile, self.rational_click_top5[:, 3], self.rational_topic_top5[:, 3],
                               self.hist_click, self.hist_topic, self.sl_hist_click)
            logit_4 = ui_scorer(self.user_profile, self.rational_click_top5[:, 4], self.rational_topic_top5[:, 4],
                               self.hist_click, self.hist_topic, self.sl_hist_click)
            logit_0 = tf.reshape(logit_0, [-1, 1])
            logit_1 = tf.reshape(logit_1, [-1, 1])
            logit_2 = tf.reshape(logit_2, [-1, 1])
            logit_3 = tf.reshape(logit_3, [-1, 1])
            logit_4 = tf.reshape(logit_4, [-1, 1])
            logit = tf.concat([logit_0, logit_1, logit_2, logit_3, logit_4], axis=1)
            rational_top5_sl = tf.clip_by_value(self.rational_sl, 0, 5)
            query, key = attention_generate(self.rational_click_top5, self.rational_topic_top5)
            self.logits_topk_positive, _, _, _ = attention_w_similarity(query, key, rational_top5_sl, logit)



        with tf.name_scope('constrative_learning'):
            def contrastive_learning_logit(attention, logit, similarity, keys):
                sample_neg = tf.TensorArray(
                    dtype=tf.float32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False
                )
                sample_aug = tf.TensorArray(
                    dtype=tf.float32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False
                )
                cl_vector_loss = tf.TensorArray(
                    dtype=tf.float32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False
                )
                init_loop_var = (
                0, logit, sample_aug, sample_neg, attention, self.sl_hist_click, similarity, keys, cl_vector_loss)

                def condition(step, logit, sample_aug, sample_neg, attention, sl, similarity, keys, cl_vector_loss):
                    return tf.less(step, tf.shape(logit)[0])

                def body(step, logit, sample_aug, sample_neg, attention, sl, similarity, keys, cl_vector_loss):
                    pos = logit[step]
                    aug_topk = tf.nn.top_k(attention[step], tf.cast(
                        tf.floor(self.aug_weight * tf.cast(tf.shape(logit)[1], dtype=tf.float32)), dtype=tf.int32))
                    aug_index = aug_topk.indices
                    aug_index = tf.cast(aug_index, tf.int32)
                    ones_index = tf.ones_like(aug_index)
                    aug_index = tf.expand_dims(aug_index, axis=1)
                    aug_scatter = tf.cast(tf.scatter_nd(aug_index, ones_index, [tf.shape(logit)[1]]), dtype=tf.float32)
                    aug_scatter = tf.where(tf.equal(aug_scatter, 0), tf.zeros_like(aug_scatter),
                                           tf.ones_like(aug_scatter))
                    aug = pos * aug_scatter
                    key_masks = tf.sequence_mask(sl[step] - 1, tf.shape(logit)[1], dtype=tf.float32) * aug_scatter
                    paddings = tf.ones_like(key_masks) * (-2 ** 32 + 1)
                    outputs_1 = tf.where(tf.cast(key_masks, dtype=tf.bool), similarity[step], paddings)
                    outputs_2 = outputs_1 / (keys[step].get_shape().as_list()[-1] ** 0.5)
                    outputs_3 = tf.nn.softmax(outputs_2)
                    aug_logit = tf.reduce_sum(tf.multiply(outputs_3, aug))

                    neg_topk = tf.nn.top_k(attention[step], tf.cast(
                        tf.floor(self.neg_weight * tf.cast(tf.shape(logit)[1], dtype=tf.float32)), dtype=tf.int32))
                    neg_index = neg_topk.indices
                    neg_index = tf.cast(neg_index, tf.int32)
                    ones_neg_index = tf.ones_like(neg_index)
                    neg_index = tf.expand_dims(neg_index, axis=1)
                    neg_scatter = tf.cast(tf.scatter_nd(neg_index, ones_neg_index, [tf.shape(logit)[1]]),
                                          dtype=tf.float32)
                    neg_scatter = tf.where(tf.equal(neg_scatter, 0), tf.zeros_like(neg_scatter),
                                           tf.ones_like(neg_scatter))
                    neg_scatter = tf.logical_not(tf.cast(neg_scatter, dtype=tf.bool))
                    neg = pos * tf.cast(neg_scatter, dtype=tf.float32)
                    key_neg_masks = tf.sequence_mask(sl[step] - 1, tf.shape(logit)[1], dtype=tf.float32) * tf.cast(
                        neg_scatter, dtype=tf.float32)
                    paddings_neg = tf.ones_like(key_neg_masks) * (-2 ** 32 + 1)
                    outputs_1_neg = tf.where(tf.cast(key_neg_masks, dtype=tf.bool), similarity[step], paddings_neg)
                    outputs_2_neg = outputs_1_neg / (keys[step].get_shape().as_list()[-1] ** 0.5)
                    outputs_3_neg = tf.nn.softmax(outputs_2_neg)
                    neg_logit = tf.reduce_sum(tf.multiply(outputs_3_neg, neg))
                    sample_aug = sample_aug.write(step, aug_logit)
                    sample_neg = sample_neg.write(step, neg_logit)
                    return step + 1, logit, sample_aug, sample_neg, attention, sl, similarity, keys, cl_vector_loss

                step_v1, logit_v1, sample_aug_v1, sample_neg_v1, attention_v1, sl_v1, similarity_v1, keys_v1, cl_vector_loss_v1 = tf.while_loop(
                    cond=condition, body=body, loop_vars=init_loop_var)
                return sample_aug_v1.stack(), sample_neg_v1.stack(), cl_vector_loss_v1.stack()

            self.aug_logits, self.neg_logits, self.cl_loss_vector = contrastive_learning_logit(self.att_hist,
                                                                                               self.score_ui,
                                                                                               self.similarity,
                                                                                               self.keys)
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        with tf.name_scope('train_op'):
            def kl_for_log_prob(log_p, log_q):
                p = tf.exp(log_p)
                neg_ent = tf.reduce_sum(p * log_p, axis=-1)
                neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
                kl = neg_ent - neg_cross_ent
                return kl

            self.logits_attention = self.logits_click
            self.logits_sigmoid = tf.sigmoid(self.logits_attention)
            self.sup = tf.ones([tf.shape(self.supervised_signals_click)[0], ], tf.float32)

            self.loss_mse_attention = tf.compat.v1.losses.mean_squared_error(labels=self.y, predictions=self.logits_attention,
                                                                   weights=((self.sup) * (
                                                                           self.sup + 0.3 * self.supervised_signals_click + 1.6 * self.supervised_signals_impression)))
            self.cl_loss = tf.maximum(0.0, tf.abs(self.logits_click - self.aug_logits) - tf.abs(
                self.logits_click - self.neg_logits) + self.margin)

            # positive
            self.loss_mse_topk_positive = tf.compat.v1.losses.mean_squared_error(labels=self.y,
                                                                       predictions=self.logits_topk_positive, weights=(
                            (self.sup) * (self.sup + 0.04 * self.supervised_signals_click + 0.04 * self.supervised_signals_impression)))
            self.kl_loss_positive = kl_for_log_prob(tf.nn.log_softmax(self.logits_attention),
                                                    tf.nn.log_softmax(self.logits_topk_positive)) + kl_for_log_prob(
                tf.nn.log_softmax(self.logits_topk_positive), tf.nn.log_softmax(self.logits_attention))
            self.loss_positive = tf.reduce_mean(
                self.loss_mse_attention + self.loss_mse_topk_positive + self.cl_loss + 100 * self.kl_loss_positive)
            trainable_params = tf.trainable_variables()
            self.opt_positive = tf.compat.v1.train.AdamOptimizer(self.lr)
            self.gradients_positive = tf.gradients(self.loss_positive, trainable_params)
            clip_gradients_positive, _ = tf.clip_by_global_norm(self.gradients_positive, 5)
            self.train_op_positive = self.opt_positive.apply_gradients(zip(clip_gradients_positive, trainable_params),
                                                                       global_step=self.global_step)
            # negative
            self.loss_negative = tf.reduce_mean(self.loss_mse_attention + self.cl_loss)
            self.opt_negative = tf.compat.v1.train.AdamOptimizer(self.lr)
            self.gradients_negative = tf.gradients(self.loss_negative, trainable_params)
            clip_gradients_negative, _ = tf.clip_by_global_norm(self.gradients_negative, 5)
            self.train_op_negative = self.opt_negative.apply_gradients(zip(clip_gradients_negative, trainable_params),
                                                                       global_step=self.global_step)
        with tf.name_scope('basic_info'):
            self.saver = tf.compat.v1.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.local_init_op = tf.local_variables_initializer()
    def train_offline(self, sess, uij, l, is_positive=False):
        if is_positive is True:
            label, loss, _ = sess.run([self.logits_sigmoid,self.loss_positive, self.train_op_positive], feed_dict={
                self.user: uij[0],
                self.sex: uij[1],
                self.province: uij[2],
                self.city: uij[3],

                self.hist_click: uij[4],
                self.hist_topic: uij[5],
                self.sl_hist_click: uij[6],
                self.long_term_click: uij[7],
                self.long_term_topic: uij[8],
                self.long_term_sl: uij[9],
                self.short_term_click: uij[10],
                self.short_term_topic: uij[11],
                self.short_term_sl: uij[12],

                self.y: uij[13],
                self.supervised_signals_click: uij[15],
                self.supervised_signals_impression: uij[16],

                # Here, we select rationale from users' short-term behaviors
                self.rational_click:uij[10],
                self.rational_topic:uij[11],
                self.rational_sl:uij[12],

                self.future_click: uij[17],
                self.future_topic: uij[18],
                self.sl_future_click: uij[19],
                self.lr: l
            })
        else:
            label,loss, _ = sess.run([self.logits_sigmoid,self.loss_negative, self.train_op_negative], feed_dict={
                self.user: uij[0],
                self.sex: uij[1],
                self.province: uij[2],
                self.city: uij[3],
                self.hist_click: uij[4],
                self.hist_topic: uij[5],
                self.sl_hist_click: uij[6],
                self.long_term_click: uij[7],
                self.long_term_topic: uij[8],
                self.long_term_sl: uij[9],
                self.short_term_click: uij[10],
                self.short_term_topic: uij[11],
                self.short_term_sl: uij[12],
                self.y: uij[13],
                self.supervised_signals_click: uij[15],
                self.supervised_signals_impression: uij[16],
                self.lr: l
            })
        return label,loss
    def eval_offline_test(self, sess, uij):
        retention_label = sess.run(
            [self.logits_sigmoid], feed_dict={
            self.user: uij[0],
            self.sex:uij[1],
            self.province:uij[2],
            self.city:uij[3],
            self.hist_click: uij[4],
            self.hist_topic:uij[5],
            self.sl_hist_click: uij[6],
            self.long_term_click: uij[7],
            self.long_term_topic: uij[8],
            self.long_term_sl: uij[9],
            self.short_term_click: uij[10],
            self.short_term_topic: uij[11],
            self.short_term_sl: uij[12],
            self.y: uij[13]
        })
        return retention_label


