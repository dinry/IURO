import numpy as np
import pdb


class DataInputTrain:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        user_positive, sex_positive, province_positive, city_positive, label_positive, label_auc_positive = [], [], [], [], [], []
        supervised_signals_click_positive, supervised_signals_impression_positive = [], []
        sl_hist_item_positive, sl_future_clicks = [], []
        sl_short_term_item_positive = []

        user_negative, sex_negative, province_negative, city_negative, label_negative, label_auc_negative = [], [], [], [], [], []
        supervised_signals_click_negative, supervised_signals_impression_negative = [], []
        sl_hist_item_negative = []
        sl_short_term_item_negative = []

        t_positive, t_negative = [], []

        for t in ts:
            # have clicks in the future (next three days)
            if len(t[11]) > 0:
                t_positive.append(t)
                user_positive.append(t[0])
                sex_positive.append(t[1])
                province_positive.append(t[2])
                city_positive.append(t[3])
                sl_hist_item_positive.append(len(t[4]))
                sl_short_term_item_positive.append(len(t[6]))
                label_positive.append(t[8])
                if t[8] == 0.0:
                    label_auc_positive.append(0)
                elif t[8] > 0.0:
                    label_auc_positive.append(1)
                supervised_signals_click_positive.append(t[9])
                supervised_signals_impression_positive.append(t[10])
                sl_future_clicks.append(len(t[11]))
            else:
                t_negative.append(t)
                user_negative.append(t[0])
                sex_negative.append(t[1])
                province_negative.append(t[2])
                city_negative.append(t[3])
                sl_hist_item_negative.append(len(t[4]))
                sl_short_term_item_negative.append(len(t[6]))
                label_negative.append(t[8])
                if t[8] == 0.0:
                    label_auc_negative.append(0)
                elif t[8] > 0.0:
                    label_auc_negative.append(1)
                supervised_signals_click_negative.append(t[9])
                supervised_signals_impression_negative.append(t[10])

        # data process for users with clicks in the next three days
        max_sl_hist_item_positive = max(sl_hist_item_positive)
        hist_item_positive = np.zeros([len(t_positive), max_sl_hist_item_positive], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[4])):
                hist_item_positive[k][l] = t[4][l]
            k += 1

        hist_topic_positive = np.zeros([len(t_positive), max_sl_hist_item_positive], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[5])):
                hist_topic_positive[k][l] = t[5][l]
            k += 1

        max_sl_short_term_item_positive = max(sl_short_term_item_positive)
        short_term_item_positive = np.zeros([len(t_positive), max_sl_short_term_item_positive], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[6])):
                short_term_item_positive[k][l] = t[6][l]
            k += 1

        short_term_topic_positive = np.zeros([len(t_positive), max_sl_short_term_item_positive], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[7])):
                short_term_topic_positive[k][l] = t[7][l]
            k += 1

        # future clicks for rational learning
        max_sl_future_click = max(sl_future_clicks)
        future_click = np.zeros([len(ts), max_sl_future_click], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[11])):
                future_click[k][l] = t[11][l]
            k += 1

        future_topic = np.zeros([len(ts), max_sl_future_click], np.int64)
        k = 0
        for t in t_positive:
            for l in range(len(t[12])):
                future_topic[k][l] = t[12][l]
            k += 1

        # data process for users without clicks in the next three days
        max_sl_hist_item_negative = max(sl_hist_item_negative)
        hist_item_negative = np.zeros([len(t_negative), max_sl_hist_item_negative], np.int64)
        k = 0
        for t in t_negative:
            for l in range(len(t[4])):
                hist_item_negative[k][l] = t[4][l]
            k += 1

        hist_topic_negative = np.zeros([len(t_negative), max_sl_hist_item_negative], np.int64)
        k = 0
        for t in t_negative:
            for l in range(len(t[5])):
                hist_topic_negative[k][l] = t[5][l]
            k += 1

        max_sl_short_term_item_negative = max(sl_short_term_item_negative)
        short_term_item_negative = np.zeros([len(t_negative), max_sl_short_term_item_negative], np.int64)
        k = 0
        for t in t_negative:
            for l in range(len(t[6])):
                short_term_item_negative[k][l] = t[6][l]
            k += 1

        short_term_topic_negative = np.zeros([len(t_negative), max_sl_short_term_item_negative], np.int64)
        k = 0
        for t in t_negative:
            for l in range(len(t[7])):
                short_term_topic_negative[k][l] = t[7][l]
            k += 1

        positive_feature = [user_positive, sex_positive, province_positive, city_positive, hist_item_positive,
                            hist_topic_positive, sl_hist_item_positive, short_term_item_positive,
                            short_term_topic_positive, sl_short_term_item_positive, label_positive, label_auc_positive,
                            supervised_signals_click_positive, supervised_signals_impression_positive, future_click,
                            future_click_topic, sl_future_clicks]
        negative_feature = [user_negative, sex_negative, province_negative, city_negative, hist_item_negative,
                            hist_topic_negative, sl_hist_item_negative, short_term_item_negative,
                            short_term_topic_negative, sl_short_term_item_negative, label_negative, label_auc_negative,
                            supervised_signals_click_negative, supervised_signals_impression_negative]
        return self.i, (positive_feature, negative_feature)


class DataInputTest:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        user, label, label_auc = [], [], []
        supervised_signals_click, supervised_signals_impression = [], []
        sex, province, city = [], [], []
        sl_hist_item, sl_short_term_clicks = [], []
        for t in ts:
            # user profile
            user.append(t[0])
            sex.append(t[1])
            province.append(t[2])
            city.append(t[3])

            # sequence length of items (historical and short term)
            sl_hist_item.append(len(t[4]))
            sl_short_term_clicks.append(len(t[6]))

            label.append(t[8])

            if t[8] == 0.0:
                label_auc.append(0)
            elif t[8] > 0.0:
                label_auc.append(1)

            # retention supervised signals
            supervised_signals_click.append(t[9])
            supervised_signals_impression.append(t[10])

        # The historical items clicked.
        max_sl_hist_item = max(sl_hist_item)
        hist_item = np.zeros([len(ts), max_sl_hist_item], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[4])):
                hist_item[k][l] = t[4][l]
            k += 1

        # The topic of historical items clicked.
        hist_topic = np.zeros([len(ts), max_sl_hist_item], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[5])):
                hist_topic[k][l] = t[5][l]
            k += 1

        # short term items clicked
        max_sl_short_term_click = max(sl_short_term_clicks)
        short_term_click = np.zeros([len(ts), max_sl_short_term_click], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[6])):
                short_term_click[k][l] = t[6][l]
            k += 1

        # short term items clicked (topic)
        short_term_click_topic = np.zeros([len(ts), max_sl_short_term_click], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[7])):
                short_term_click_topic[k][l] = t[7][l]
            k += 1

        return self.i, (
            user, sex, province, city, hist_item, hist_topic, sl_hist_item, short_term_click, short_term_click_topic,
            sl_short_term_clicks, label, label_auc,
            supervised_signals_click, supervised_signals_impression)
