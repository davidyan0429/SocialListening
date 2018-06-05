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
    for sent in text:
        samples.append(re.split('[，,。！!？?:：；;\s]', sent))
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

def predict(text, dict, model, ID2label):
    seq = text_to_sequence(text, dict)
    sequences = []
    sequences.append(seq)
    X = pad_sequences(sequences, maxlen=100)
    preds = model.predict(X, batch_size=10000, verbose=0)
    labels_pred = probs2label(preds)
    label = ID2label[labels_pred[0]]
    return label

def batch_predict(text):
    samples, ID2label_class, ID2label_sentiment = load_data(text)
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
    for sents in samples:
        selected_sent = []
        sent_result = []
        for i in range(len(sents)):
            predict_label_class2_1 = predict(sents[i], dict_class, model_class1, ID2label_class)
            predict_label_class2_2 = predict(sents[i], dict_class, model_class2, ID2label_class)
            predict_label_class2_3 = predict(sents[i], dict_class, model_class3, ID2label_class)
            predict_label_class2_4 = predict(sents[i], dict_class, model_class4, ID2label_class)
            predict_label_class2_5 = predict(sents[i], dict_class, model_class5, ID2label_class)
            predict_label_class2_6 = predict(sents[i], dict_class, model_class6, ID2label_class)
            predict_label_class2_7 = predict(sents[i], dict_class, model_class7, ID2label_class)
            predict_label_class2_8 = predict(sents[i], dict_class, model_class8, ID2label_class)
            predict_label_class2_9 = predict(sents[i], dict_class, model_class9, ID2label_class)
            predict_label_class2_10 = predict(sents[i], dict_class, model_class10, ID2label_class)
            predict_label_sentiment1 = predict(sents[i], dict_sentiment, model_sentiment1, ID2label_sentiment)
            predict_label_sentiment2 = predict(sents[i], dict_sentiment, model_sentiment2, ID2label_sentiment)
            predict_label_sentiment3 = predict(sents[i], dict_sentiment, model_sentiment3, ID2label_sentiment)
            predict_label_sentiment4 = predict(sents[i], dict_sentiment, model_sentiment4, ID2label_sentiment)
            predict_label_sentiment5 = predict(sents[i], dict_sentiment, model_sentiment5, ID2label_sentiment)

            predict_list_class = [predict_label_class2_1, predict_label_class2_2, predict_label_class2_3, predict_label_class2_4, predict_label_class2_5,
                            predict_label_class2_6, predict_label_class2_7, predict_label_class2_8, predict_label_class2_9, predict_label_class2_10]
            predict_label_class2 = max(predict_list_class, key=predict_list_class.count)
            class2_class1 = load_class2_to_class1()
            predict_label_class1 = class2_class1[predict_label_class2]

            predict_list_senti = [predict_label_sentiment1, predict_label_sentiment2, predict_label_sentiment3,
                            predict_label_sentiment4, predict_label_sentiment5]
            predict_label_sentiment = max(predict_list_senti, key=predict_list_senti.count)
            if predict_label_class1 != '其他' and predict_label_sentiment != '中性':
                selected_sent.append(sents[i])
                sent_result.append(str((predict_label_class1, predict_label_class2, predict_label_sentiment)))
                #print(sents[i]+'\t'+predict_label_class1+'\t'+predict_label_class2+'\t'+predict_label_sentiment)
        result.append('，'.join(selected_sent)+'\t'+', '.join(sent_result))
    print("predict:"+str(time.clock() - start))
    writer = codecs.open('./data/result.txt', "w", encoding='utf-8',errors='ignore')
    for text in result:
        writer.write(text+'\n')
    writer.flush()
    writer.close()

if __name__ == '__main__':
    test = []
    file = codecs.open('./data/test.txt', "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        test.append(line)
    #test = ['客服还不错？用的很舒服，但是电池不耐用。还有点贵！',
    #         '高配CPU!如果价格低一些就更好了',
    #         '带手写功能的平板用着的确很舒服,就是接口太少了',
    #         '发热厉害!!但是看着很高档']
    batch_predict(test)