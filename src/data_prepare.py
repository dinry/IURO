import pdb
import pickle
import random
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
def dataset_7_3_split(datafile_path):
    f6 = open("zhihu1M_7_3_split.txt", 'w')
    history_length = []
    index_seven_three = 0

    for line in open(datafile_path, "r", encoding='utf-8'):
        ss = line[:-1].split('\t')
        cc = ss[2].split(',')
        item_seven_days = []
        item_three_days = []
        for i in range(len(cc)):
            ww = cc[i].split('|')
            if int(ww[1]) > 1525279257 + 7 * 24 * 60 * 60:
                item_three_days.append(cc[i])
            else:
                item_seven_days.append(cc[i])
        str = ','
        if len(item_seven_days) != 0:
            if len(item_three_days) == 0:
                index_seven_three = index_seven_three + 1
            history_length.append(len(item_seven_days))
            f6.write(ss[0] + "\t" + str.join(item_seven_days) + "\t" + str.join(item_three_days))
            f6.write("\n")
    return "zhihu1M_7_3_split.txt"
def index_generate(interaction_file_path,answer_info_file,user_info_file):
    dict_user={}
    dict_sex={}
    dict_province={}
    dict_city={}
    dict_answer={}
    dict_topic={}
    dict_author={}
    user_count=0
    sex_count=0
    province_count=0
    city_count=0
    answer_count=0
    topic_count=0
    author_count=0
    #pdb.set_trace()
    for line in open(interaction_file_path,"r",encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        if line[0] not in dict_user.keys():
            user_count = user_count + 1
            dict_user[line[0]] = user_count
            # user_count = user_count + 1
        answers = line[1].split(',')
        for answer in answers:
            if answer.split('|')[0] not in dict_answer.keys():
                answer_count = answer_count + 1
                dict_answer[answer.split('|')[0]] = answer_count
                # answer_count = answer_count + 1
    #pdb.set_trace()
    for line in open(user_info_file,"r",encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        if line[0] in dict_user.keys():
            #pdb.set_trace()
            sex=line[2]
            province=line[-3]
            city=line[-2]
            if sex not in dict_sex.keys():
                sex_count=sex_count+1
                dict_sex[sex]=sex_count
                # sex_count = sex_count + 1
            if province not in dict_province.keys():
                province_count=province_count+1
                dict_province[province]=province_count
                # province_count = province_count + 1
            if city not in dict_city.keys():
                city_count=city_count+1
                dict_city[city]=city_count
                # city_count = city_count + 1
    #pdb.set_trace()

    for line in open(answer_info_file, "r", encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        #pdb.set_trace()
        if line[0] in dict_answer.keys():
            # pdb.set_trace()
            topics=line[17].strip(' ')
            topics=topics[:-1].split(',')
            for topic in topics:
                if topic not in dict_topic.keys():
                    topic_count = topic_count + 1
                    dict_topic[topic] = topic_count
                    # topic_count = topic_count + 1
        #pdb.set_trace()
        if line[0] in dict_answer.keys():
            #pdb.set_trace()
            author= line[3].split(',')[0]
            if author not in dict_author.keys():
                author_count=author_count+1
                dict_author[author]=author_count
                # author_count = author_count + 1
        #pdb.set_trace()
    return dict_user,dict_sex,dict_province,dict_city,dict_answer,dict_topic,dict_author,user_count,sex_count,province_count,city_count,answer_count,topic_count,author_count


def delete_answer_without_train_in_test(file_path,dict_answer):
    write_path="zhihu1M_7_3_split_without_train_in_test.txt"
    write = open(write_path, 'w')
    for line in open(file_path, "r", encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        test_items = line[2].split(',')
        if len(line[2].split(',')[-1].split('|')) != 1:
            answers = line[2].split(',')
            for answer in answers[::-1]:
                if answer.split('|')[0] not in dict_answer.keys():
                    answers.remove(answer)
            test_items = answers
        str = ','
        write.write(line[0] + "\t" + line[1] + "\t" + str.join(test_items))
        write.write("\n")
        write.flush()
    return write_path
def user_infos_generate(dict_user,dict_sex,dict_province,dict_city,user_info_file):
    user_infos={}
    for line in open(user_info_file,"r",encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        if line[0] in dict_user.keys():
            sex=line[2]
            province=line[-3]
            city=line[-2]
            user_infos[dict_user[line[0]]]=[dict_sex[sex],dict_province[province],dict_city[city]]
    return user_infos
def answer_infos_generate(dict_answer,dict_topic,dict_author,answer_info_file):
    answer_infos={}
    for line in open(answer_info_file,"r"):
        line=line[:-1].strip('\n').split('\t')
        if line[0] in dict_answer.keys():
            author=line[3]
            topics=line[17].strip(' ')
            topics=topics[:-1].split(',')
            t=[]
            for topic in topics:
                t.append(dict_topic[topic])
            answer_infos[dict_answer[line[0]]]=[t,dict_author[author]]
    return answer_infos

def liucun_label_generate(first_day_items,second_day_items,third_day_items,first_day_clicks,second_day_clicks,third_day_clicks):
    liucun=[]
    ctr_info=[]
    # supervised signals
    max_click_count=1

    for i in range(len(first_day_items)):
        label_click_1=0
        label_click_2=0
        lable_click_3=0
        label_1 = 0
        label_2 = 0
        label_3 = 0
        first_index_impression = np.where(first_day_clicks[i, 0:len(first_day_clicks[0])] != -1)[0].tolist()
        second_index_impression = np.where(second_day_clicks[i, 0:len(second_day_clicks[0])] != -1)[0].tolist()
        third_index_impression = np.where(third_day_clicks[i, 0:len(third_day_clicks[0])] != -1)[0].tolist()
        if len(first_index_impression)>0:
            label_1=1
        if len(second_index_impression)>0:
            label_2=1
        if len(third_index_impression)>0:
            label_3=1
        label=label_1+label_2+label_3
        liucun.append(label)
        first_index_click = np.where(first_day_clicks[i, 0:len(first_day_clicks[0])] == 1)[0].tolist()
        second_index_click = np.where(second_day_clicks[i, 0:len(second_day_clicks[0])] == 1)[0].tolist()
        third_index_click = np.where(third_day_clicks[i, 0:len(third_day_clicks[0])] == 1)[0].tolist()
        if len(first_index_click)>0:
            label_click_1=len(first_index_click)
        if len(second_index_click)>0:
            label_click_2=len(second_index_click)
        if len(third_index_click)>0:
            lable_click_3=len(third_index_click)
        click_count=label_click_1+label_click_2+lable_click_3
        ctr_info.append(math.log(1+click_count,math.e))
    return liucun,ctr_info
def liucun_label_generate_test(first_day_items,second_day_items,third_day_items,first_day_clicks,second_day_clicks,third_day_clicks):
    liucun=[]
    ctr_info=[]
    for i in range(len(first_day_items)):
        label_click_1=0
        label_click_2=0
        lable_click_3=0
        label_1 = 0
        label_2 = 0
        label_3 = 0
        pdb.set_trace()
        first_index_impression = np.where(first_day_clicks[i, 0:len(first_day_clicks[0])] != -1)[0].tolist()
        second_index_impression = np.where(second_day_clicks[i, 0:len(second_day_clicks[0])] != -1)[0].tolist()
        third_index_impression = np.where(third_day_clicks[i, 0:len(third_day_clicks[0])] != -1)[0].tolist()
        if len(first_index_impression)>0:
            label_1=1
        if len(second_index_impression)>0:
            label_2=1
        if len(third_index_impression)>0:
            label_3=1
        label=label_1+label_2+label_3
        liucun.append(label)
        first_index_click = np.where(first_day_clicks[i, 0:len(first_day_clicks[0])] == 1)[0].tolist()
        second_index_click = np.where(second_day_clicks[i, 0:len(second_day_clicks[0])] == 1)[0].tolist()
        third_index_click = np.where(third_day_clicks[i, 0:len(third_day_clicks[0])] == 1)[0].tolist()
        if len(first_index_click)>0:
            label_click_1=1
        if len(second_index_click)>0:
            label_click_2=1
        if len(third_index_click)>0:
            lable_click_3=1
        ctr_info.append((label_click_1+label_click_2+lable_click_3)/3.0)
    return liucun,ctr_info
def data_generate(file_path,dict_u,dict_i,user_infos,answer_infos):
    user_list = []

    one_item_lists=[]
    one_item_click_lists = []

    two_item_lists=[]
    two_item_click_lists = []

    three_item_lists=[]
    three_item_click_lists = []

    four_item_lists=[]
    four_item_click_lists = []

    five_item_lists=[]
    five_item_click_lists = []

    six_item_lists=[]
    six_item_click_lists = []

    seven_item_lists=[]
    seven_item_click_lists = []

    eight_item_lists=[]
    eight_item_click_lists = []

    nine_item_lists=[]
    nine_item_click_lists = []

    ten_item_lists=[]
    ten_item_click_lists = []
    for line in open(file_path, "r", encoding='utf-8'):
        line = line[:-1].strip('\n').split('\t')
        user_list.append(dict_u[line[0]])

        one_item_list = []
        one_item_click_list = []

        two_item_list = []
        two_item_click_list = []

        three_item_list = []
        three_item_click_list = []

        four_item_list = []
        four_item_click_list = []

        five_item_list = []
        five_item_click_list = []

        six_item_list = []
        six_item_click_list = []

        seven_item_list = []
        seven_item_click_list = []

        eight_item_list = []
        eight_item_click_list = []

        nine_item_list = []
        nine_item_click_list = []

        ten_item_list = []
        ten_item_click_list = []

        cc=line[1].split(',')
        for i in range(len(cc)):
            ww = cc[i].split('|')
            if len(ww) == 1:
                break
            if int(ww[1]) > 1525279257 and int(ww[1]) <= 1525279257 + 1 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    one_item_list.append(dict_i[ww[0]])
                    one_item_click_list.append(1)
                elif int(ww[2])==0:
                    one_item_list.append(dict_i[ww[0]])
                    one_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 1 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 2 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    two_item_list.append(dict_i[ww[0]])
                    two_item_click_list.append(1)
                else:
                    two_item_list.append(dict_i[ww[0]])
                    two_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 2 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 3 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    three_item_list.append(dict_i[ww[0]])
                    three_item_click_list.append(1)
                else:
                    three_item_list.append(dict_i[ww[0]])
                    three_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 3 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 4 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    four_item_list.append(dict_i[ww[0]])
                    four_item_click_list.append(1)
                else:
                    four_item_list.append(dict_i[ww[0]])
                    four_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 4 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 5 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    five_item_list.append(dict_i[ww[0]])
                    five_item_click_list.append(1)
                else:
                    five_item_list.append(dict_i[ww[0]])
                    five_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 5 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 6 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    six_item_list.append(dict_i[ww[0]])
                    six_item_click_list.append(1)
                else:
                    six_item_list.append(dict_i[ww[0]])
                    six_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 6 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 7 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    seven_item_list.append(dict_i[ww[0]])
                    seven_item_click_list.append(1)
                else:
                    seven_item_list.append(dict_i[ww[0]])
                    seven_item_click_list.append(0)


        cc = line[2].split(',')
        for i in range(len(cc)):
            ww = cc[i].split('|')
            if len(ww) == 1:
                break
            if int(ww[1]) > 1525279257 + 7 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 8 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    eight_item_list.append(dict_i[ww[0]])
                    eight_item_click_list.append(1)
                else:
                    eight_item_list.append(dict_i[ww[0]])
                    eight_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 8 * 24 * 60 * 60 and int(ww[1]) <= 1525279257 + 9 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    nine_item_list.append(dict_i[ww[0]])
                    nine_item_click_list.append(1)
                else:
                    nine_item_list.append(dict_i[ww[0]])
                    nine_item_click_list.append(0)
            if int(ww[1]) > 1525279257 + 9 * 24 * 60 * 60:
                if int(ww[2]) > 0:
                    ten_item_list.append(dict_i[ww[0]])
                    ten_item_click_list.append(1)
                else:
                    ten_item_list.append(dict_i[ww[0]])
                    ten_item_click_list.append(0)
        one_item_lists.append(one_item_list)
        one_item_click_lists.append(one_item_click_list)
        two_item_lists.append(two_item_list)
        two_item_click_lists.append(two_item_click_list)
        three_item_lists.append(three_item_list)
        three_item_click_lists.append(three_item_click_list)
        four_item_lists.append(four_item_list)
        four_item_click_lists.append(four_item_click_list)
        five_item_lists.append(five_item_list)
        five_item_click_lists.append(five_item_click_list)
        six_item_lists.append(six_item_list)
        six_item_click_lists.append(six_item_click_list)
        seven_item_lists.append(seven_item_list)
        seven_item_click_lists.append(seven_item_click_list)
        eight_item_lists.append(eight_item_list)
        eight_item_click_lists.append(eight_item_click_list)
        nine_item_lists.append(nine_item_list)
        nine_item_click_lists.append(nine_item_click_list)
        ten_item_lists.append(ten_item_list)
        ten_item_click_lists.append(ten_item_click_list)
    user=pd.DataFrame(user_list)

    one_items=pd.DataFrame(one_item_lists)
    one_items=one_items.fillna(-1)
    one_items_click = pd.DataFrame(one_item_click_lists)
    one_items_click = one_items_click.fillna(-1)

    two_items = pd.DataFrame(two_item_lists)
    two_items = two_items.fillna(-1)
    two_items_click = pd.DataFrame(two_item_click_lists)
    two_items_click = two_items_click.fillna(-1)

    three_items=pd.DataFrame(three_item_lists)
    three_items=three_items.fillna(-1)
    three_items_click = pd.DataFrame(three_item_click_lists)
    three_items_click = three_items_click.fillna(-1)

    four_items = pd.DataFrame(four_item_lists)
    four_items = four_items.fillna(-1)
    four_items_click = pd.DataFrame(four_item_click_lists)
    four_items_click = four_items_click.fillna(-1)

    five_items=pd.DataFrame(five_item_lists)
    five_items=five_items.fillna(-1)
    five_items_click = pd.DataFrame(five_item_click_lists)
    five_items_click = five_items_click.fillna(-1)

    six_items = pd.DataFrame(six_item_lists)
    six_items = six_items.fillna(-1)
    six_items_click = pd.DataFrame(six_item_click_lists)
    six_items_click = six_items_click.fillna(-1)

    seven_items=pd.DataFrame(seven_item_lists)
    seven_items=seven_items.fillna(-1)
    seven_items_click = pd.DataFrame(seven_item_click_lists)
    seven_items_click = seven_items_click.fillna(-1)

    eight_items = pd.DataFrame(eight_item_lists)
    eight_items = eight_items.fillna(-1)
    eight_items_click = pd.DataFrame(eight_item_click_lists)
    eight_items_click = eight_items_click.fillna(-1)


    nine_items=pd.DataFrame(nine_item_lists)
    nine_items=nine_items.fillna(-1)
    nine_items_click = pd.DataFrame(nine_item_click_lists)
    nine_items_click = nine_items_click.fillna(-1)

    ten_items = pd.DataFrame(ten_item_lists)
    ten_items = ten_items.fillna(-1)
    ten_items_click = pd.DataFrame(ten_item_click_lists)
    ten_items_click = ten_items_click.fillna(-1)


    user = user.to_numpy(dtype=int)
    one_items = one_items.to_numpy(dtype=int)
    one_items_click = one_items_click.to_numpy(dtype=int)
    two_items = two_items.to_numpy(dtype=int)
    two_items_click = two_items_click.to_numpy(dtype=int)
    three_items = three_items.to_numpy(dtype=int)
    three_items_click = three_items_click.to_numpy(dtype=int)
    four_items = four_items.to_numpy(dtype=int)
    four_items_click = four_items_click.to_numpy(dtype=int)
    five_items = five_items.to_numpy(dtype=int)
    five_items_click = five_items_click.to_numpy(dtype=int)
    six_items = six_items.to_numpy(dtype=int)
    six_items_click = six_items_click.to_numpy(dtype=int)
    seven_items = seven_items.to_numpy(dtype=int)
    seven_items_click = seven_items_click.to_numpy(dtype=int)
    eight_items = eight_items.to_numpy(dtype=int)
    eight_items_click = eight_items_click.to_numpy(dtype=int)
    nine_items = nine_items.to_numpy(dtype=int)
    nine_items_click = nine_items_click.to_numpy(dtype=int)
    ten_items = ten_items.to_numpy(dtype=int)
    ten_items_click = ten_items_click.to_numpy(dtype=int)
    items_data=[one_items,two_items,three_items,four_items,five_items,six_items,seven_items,eight_items,nine_items,ten_items]
    clicks_data=[one_items_click,two_items_click,three_items_click,four_items_click,five_items_click,six_items_click,seven_items_click,eight_items_click,nine_items_click,ten_items_click]

    train_liucun=[]
    test_liucun=[]
    items=one_items
    clicks=one_items_click
    for index_column in range(1, 5):
        label_liucun,label_ctr = liucun_label_generate(items_data[index_column], items_data[index_column + 1],items_data[index_column + 1], clicks_data[index_column],clicks_data[index_column + 1], clicks_data[index_column + 2])
        for index_row in range(0, len(seven_items)):
            index_click = np.where(clicks[index_row, 0:len(clicks[0])] == 1)[0].tolist()
            if len(index_click)!=0:
                u = user[index_row].tolist()[0]
                row=items[index_row]
                train=row[index_click]
                hist_i = train.tolist()
                train_liucun.append([u, hist_i, label_liucun[index_row],label_ctr[index_row]])
        items = np.concatenate([items, items_data[index_column]], axis=1)
        clicks = np.concatenate([clicks, clicks_data[index_column]], axis=1)
    items = np.concatenate([items, items_data[5],items_data[6]], axis=1)
    clicks = np.concatenate([clicks, clicks_data[5],clicks_data[6]], axis=1)
    label_liucun,_ = liucun_label_generate(items_data[7], items_data[8],items_data[9], clicks_data[7],clicks_data[8], clicks_data[9])
    for index_row in range(0, len(items)):
        index_click = np.where(clicks[index_row, 0:len(clicks[0])] == 1)[0].tolist()
        if len(index_click)!=0:
            u = user[index_row].tolist()[0]
            row=items[index_row]
            train=row[index_click]
            hist_i = train.tolist()
            test_liucun.append([u, hist_i, label_liucun[index_row]])
    random.shuffle(test_liucun)
    return train_liucun,test_liucun
# user, sex, province, city, hist_item, hist_topic, short_term_item, short_term_topic, label, supervised_signals_clicks, supervised_signals_impressions, future_item, future_topic
