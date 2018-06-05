# -*- coding:utf-8 -*-
import os
import re
import numpy as np
import codecs
import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def load_data(text):
    classname2ID = {}
    ID2label_class = {}
    file = codecs.open('./data/class2_labels.txt', "r", encoding='utf-8', errors='ignore')
    count = 0
    for line in file.readlines():
        classname2ID[line.strip()] = count
        ID2label_class[count] = line.strip()
        count += 1
    ID2label_senti = {}
    ID2label_senti[0] = '中性'
    ID2label_senti[1] = '正向'
    ID2label_senti[2] = '负向'
    samples = []
    sum=0
    for sent in text:
        sent_list=re.split('[，,。！!？?:：；;]', sent)
        while "" in sent_list:
            sent_list.remove("")
        samples.append(sent_list)
        sum = sum+len(sent_list)
    print("# short sentence:", sum)
    return samples, ID2label_class, ID2label_senti

def text_to_sequence(text, dict):
    seq = []
    for char in text:
        if char in dict:
            seq.append(dict[char])
        else:
            seq.append(dict['UNK'])
    return seq

def load_dict(path):
    file = codecs.open(path, "r", encoding='utf-8',errors='ignore')
    dict = {}
    for line in file.readlines():
        line = line.strip()
        dict[line] = len(dict)
    return dict

def probs2label(pred):
    labels = []
    for probs in pred:
        probs = np.asarray(probs)
        index = np.argmax(probs)
        labels.append(index)
    return labels

def load_class2_to_class1():
    file = codecs.open('./data/class2_class1.txt', "r", encoding='utf-8', errors='ignore')
    class2_class1 = {}
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        class2_class1[terms[0]] = terms[1]
    return class2_class1

# def predict(text, dict, model, ID2label):
#     seq = text_to_sequence(text, dict)
#     sequences = []
#     sequences.append(seq)
#     X = pad_sequences(sequences, maxlen=100)
#     preds = model.predict(X, batch_size=10000, verbose=0)
#     print(preds)
#     labels_pred = probs2label(preds)
#     print(labels_pred)
#     label = ID2label[labels_pred[0]]
#     print(label)
#     return label

def predict(texts, dict, model, ID2label):
    sen2text = []
    textID = 0
    sens_all = []
    for text in texts:
        textID += 1
        # sens = split_long_sentence(text)
        for sen in text:
            sen2text.append(textID)
            sens_all.append(sen)
    sequences = []
    for sen in sens_all:
        seq = text_to_sequence(sen, dict)
        sequences.append(seq)
    X = pad_sequences(sequences, maxlen=100)
    preds = model.predict(X, batch_size=500, verbose=0)
    # print("preds:", preds)
    # print("pl:", len(preds))
    labels_pred = probs2label(preds)
    # print("labels_pred:", labels_pred)
    labels = []
    label_temp = []
    for i in range(len(labels_pred)):
        sen = sens_all[i]
        pred = labels_pred[i]
        # print("pred:", pred)
        # pred = pred[len(pred) - len(sen):]
        # label = probs2label(pred)
        label = ID2label[pred]
        # print("label:", label)
        if i == 0:
            label_temp.append(label)
        elif sen2text[i] == sen2text[i - 1]:
            label_temp.append(label)
        else:
            labels.append(label_temp)
            label_temp = []
            label_temp.append(label)
        if (i + 1) == len(labels_pred):
            labels.append(label_temp)
    return labels

def batch_predict(text):
    samples, ID2label_class, ID2label_sentiment = load_data(text)
    print("# long sentence", samples)
    load_start = time.clock()
    dict_class = load_dict('./code/class_model/class2_weight_dropout0.2.dict')
    dict_sentiment = load_dict('./code/sentiment_model/sentiment_weight_no_reduce_produce.dict')

    model_class1 = load_model('./code/class_model/model_class2_dropout0.2_1.h5')
    model_class2 = load_model('./code/class_model/model_class2_dropout0.2_2.h5')
    model_class3 = load_model('./code/class_model/model_class2_dropout0.2_3.h5')
    model_class4 = load_model('./code/class_model/model_class2_dropout0.2_4.h5')
    model_class5 = load_model('./code/class_model/model_class2_dropout0.2_5.h5')
    model_class6 = load_model('./code/class_model/model_class2_weight_new6.h5')
    model_class7 = load_model('./code/class_model/model_class2_weight_new7.h5')
    model_class8 = load_model('./code/class_model/model_class2_weight_new8.h5')
    model_class9 = load_model('./code/class_model/model_class2_weight_new9.h5')
    model_class10 = load_model('./code/class_model/model_class2_weight_new10.h5')
    model_sentiment1 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce1.h5')
    model_sentiment2 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce2.h5')
    model_sentiment3 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce3.h5')
    model_sentiment4 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce4.h5')
    model_sentiment5 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce5.h5')
    print("load:"+str(time.clock() - load_start))
    result = []
    start = time.clock()
    # for sents in samples:
    #     selected_sent = []
    #     sent_result = []
    #     for i in range(len(sents)):
    predict_label_class2_1 = predict(samples, dict_class, model_class1, ID2label_class)
    predict_label_class2_2 = predict(samples, dict_class, model_class2, ID2label_class)
    predict_label_class2_3 = predict(samples, dict_class, model_class3, ID2label_class)
    predict_label_class2_4 = predict(samples, dict_class, model_class4, ID2label_class)
    predict_label_class2_5 = predict(samples, dict_class, model_class5, ID2label_class)
    predict_label_class2_6 = predict(samples, dict_class, model_class6, ID2label_class)
    predict_label_class2_7 = predict(samples, dict_class, model_class7, ID2label_class)
    predict_label_class2_8 = predict(samples, dict_class, model_class8, ID2label_class)
    predict_label_class2_9 = predict(samples, dict_class, model_class9, ID2label_class)
    predict_label_class2_10 = predict(samples, dict_class, model_class10, ID2label_class)
    predict_label_sentiment1 = predict(samples, dict_sentiment, model_sentiment1, ID2label_sentiment)
    predict_label_sentiment2 = predict(samples, dict_sentiment, model_sentiment2, ID2label_sentiment)
    predict_label_sentiment3 = predict(samples, dict_sentiment, model_sentiment3, ID2label_sentiment)
    predict_label_sentiment4 = predict(samples, dict_sentiment, model_sentiment4, ID2label_sentiment)
    predict_label_sentiment5 = predict(samples, dict_sentiment, model_sentiment5, ID2label_sentiment)
    print(predict_label_class2_1)
    class2_class1 = load_class2_to_class1()
    predict_label_class1 = []
    predict_label_class2 = []
    predict_label_sentiment = []

    for long in range(len(predict_label_class2_1)):
        inner_term_class1 = []
        inner_term_class2 = []
        inner_term_senti = []

        selected_sent = []
        sent_result = []
        for short in range(len(predict_label_class2_1[long])):
            predict_list_class = [predict_label_class2_1[long][short], predict_label_class2_2[long][short], predict_label_class2_3[long][short], predict_label_class2_4[long][short], predict_label_class2_5[long][short],
                            predict_label_class2_6[long][short], predict_label_class2_7[long][short], predict_label_class2_8[long][short], predict_label_class2_9[long][short], predict_label_class2_10[long][short]]
            predict_label_class2_term = max(predict_list_class, key=predict_list_class.count)
            predict_label_class1_term = class2_class1[predict_label_class2_term]

            predict_list_senti = [predict_label_sentiment1[long][short], predict_label_sentiment2[long][short], predict_label_sentiment3[long][short],
                    predict_label_sentiment4[long][short], predict_label_sentiment5[long][short]]
            predict_label_sentiment_term = max(predict_list_senti, key=predict_list_senti.count)
            if predict_label_class1_term != u'其他' and predict_label_sentiment_term != u'中性':
                selected_sent.append(samples[long][short])
                sent_result.append(str((predict_label_class1_term, predict_label_class2_term, predict_label_sentiment_term)))
                # inner_term_class1.append(predict_label_class1_term)
                # inner_term_class2.append(predict_label_class2_term)
                #
                # inner_term_senti.append(predict_label_sentiment_term)
        result.append(', '.join(selected_sent) + '\t' + ', '.join(sent_result))
        # predict_label_class1.append(inner_term_class1)
        # predict_label_class2.append(inner_term_class2)
        # predict_label_sentiment.append(inner_term_senti)

    # print(predict_label_class1)
    # print(predict_label_class2)
    # print(predict_label_sentiment)

    # if predict_label_class1 != '其他' and predict_label_sentiment != '中性':
    #     selected_sent.append(sents[i])
    #     sent_result.append(str((predict_label_class1, predict_label_class2, predict_label_sentiment)))
    #     # print(sents[i]+'\t'+predict_label_class1+'\t'+predict_label_class2+'\t'+predict_label_sentiment)
    # result.append('，'.join(selected_sent)+'\t'+', '.join(sent_result))
    print("predict:"+str(time.clock() - start))
    writer = codecs.open('./data/result.txt', "w", encoding='utf-8',errors='ignore')
    for text in result:
        writer.write(text+'\n')
    writer.flush()
    writer.close()

if __name__ == '__main__':
    test = []
    file = codecs.open('./data/test1.txt', "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        test.append(line)
    # for i in range(10):
    #     test.extend(test)
    file.close()
    print(len(test))
    # test = ['客服还不错？用的很舒服，但是电池不耐用。还有点贵！',
    #         '高配CPU!如果价格低一些就更好了',
    #         '带手写功能的平板用着的确很舒服,就是接口太少了',
    #         '发热厉害!!但是看着很高档']
    batch_predict(test)