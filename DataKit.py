# -*- coding: utf8 -*-
import datetime
import pytz
import configparser
import hashlib
import hmac
import base64
import requests
import json

import os
import re
import numpy as np
import codecs
import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pyodbc

class DataKit:
    
    def __init__(self):
        self.config = self.readConfiguration()
        #self.teamId = self.getTeamId()
        # Read Data from AdMaster
        self.teamId = 73
        self.RuleGroupIds = {
            "8689": "Xbox",
            "708": "Surface",
            "2926": "竞品",
            "711": "windows",
            "3081": "Office"
        }
        self.RuleIdNames = {
            # xbox
            "81081": "xpa",
            "79157": "Xbox",
            "81093": "主机游戏",
            "81082": "win10游戏",
            "81083": "X1X",
            "81088": "中国之星",
            "79158": "PS4",
            "81090": "halo",
            "81091": "光环",
            "81089": "4K",
            "81086": "Forza",
            "81087": "极限竞速",
            "81092": "独占游戏",

            # Surface
            "23333": "Surface",
            "105321": "微软平板",
            "105322": "微软苏菲",
            "105323": "苏菲平板",

            # 竞品
            "27684": "macbook",
            "105324": "mac air",
            "105325": "mac pro",
            "105326": "ipad",

            # windows
            "105254": "Windows",
            "105255": "Win10",
            "105256": "Cortana",
            "105257": "微软小娜",
            "105258": "Microsoft Edge",
            "105259": "Edge浏览器",

            # Office            
            "28381": "Office2016",
            "105327": "Office365",
            "105328": "Office 2016",
            "105329": "Office 365",
            "105330": "Office家庭",
            "105331": "预装Office",
            "28424": "WPS"
        }

        self.RuleIds = [
            # xbox
            #"81081", "79157", "81093", "81082", "81083", "81088", "79158",
            #"81090", "81091", "81089", "81086", "81087", "81092",

            # Surface
            "23333", "105321", "105322", "105323",

            # 竞品
            #"27684", "105324", "105325", "105326",

            # windows
            #"105254", "105255", "105256", "105257", "105258", "105259",

            # Office            
            #"28381", "105327", "105328", "105329", "105330", "105331", "28424"
        ]

        self.RuleGroup2RuleId = {
            "8689":[
                "81081", "79157", "81093", "81082", "81083",
                "81088", "79158", "81090", "81091", "81089",
                "81086", "81087", "81092"
            ],
            "708":[
                "23333", "105321", "105322", "105323"
            ],
            "2926":[
                "27684", "105324", "105325", "105326"
            ],
            "711":[
                "105254", "105255", "105256", "105257", "105258", "105259"
            ],
            "3081":[
                "28381", "105327", "105328", "105329", "105330",
                "105331", "28424"
            ]
        }

        self.Platforms = ["weibo", "weixin", "blog", "wenda", "news"]

        '''
        self.Cols = {
            "projectID":0,"id":1,"platform":2,"publishedAt":3,"topic":4,
            "url":5,"content":6,"source":7,"commentSentiment":8,"score":9,
            "viewCount":10,"likeCount":11,"commentCount":12,"repostCount":13,
            "interactCount":14,"haslink":15,"isOriginal":16,"postFrom":17,
            "nickName":18,"uid":19,"profileImageUrl":20,"gender":21,"province":22,
            "city":23,"description":24,"friendCount":25,"followerCount":26,
            "statusCount":27,"favouriteCount":28,"biFollowerCount":29,
            "createdAt":30,"verified":31,"verifiedType":32,"verifiedReason":33,
            "ugc":34,"pbw":35,"originalID":36,"originalPublishedAt":37,
            "originalContent":38,"originalSource":39,"originalCommentCount":40,
            "originalRepostCount":41,"originalLikeCount":42,"originalPostFrom":43,
            "originalUID":44,"originalNickName":45,"originalVerified":46,
            "isDeleted":47,"spam":48,"images":49,"originalImages":50,"rule":51,
            "rewardCount":52,"isReward":53,"title":54,"indexid":55,"author":56,
            "digest":57,"userName":58,"biz":59,"account":60,"codeImageUrl":61,
            "openid":62,"originalUrl":63,"channel":64,"secondChannel":65,
            "thirdChannel":66,"floor":67,"isComment":68,"isEssence":69,"userUrl":70,"userLevel":71
        }
        '''
        self.Cols = [
            "projectID", "id", "platform","publishedAt","topic",
            "url","content","source","commonSentiment","score",
            "viewCount","likeCount","commentCount","repostCount",
            "interactCount","haslink","isOriginal","postFrom",
            "nickName","uid","profileImageUrl","gender","province",
            "city","description","friendCount","followerCount",
            "statusCount","favouriteCount","biFollowerCount",
            "createdAt","verified","verifiedType","verifiedReason",
            "ugc","pbw","originalID","originalPublishedAt",
            "originalContent","originalSource","originalCommentCount",
            "originalRepostCount","originalLikeCount","originalPostFrom",
            "originalUID","originalNickName","originalVerified",
            "isDeleted","spam","images","originalImages","rule",
            "dataTag", "imageTag", "videoContent",
            "rewardCount","isReward","title","indexid","author",
            "digest","userName","biz","account","codeImageUrl",
            "openid","originalUrl","channel","secondChannel",
            "thirdChannel","floor","isComment","isEssence","userUrl","userLevel"
        ]
        
        self.URI = "/teams/73/analysis/contents"
        self.headers = self.getHeaders("POST", self.URI)
        self.url = self.config["ADMASTER"]["API_URL"] + self.URI

        # AI Engine
        self.dict_class = None
        self.dict_sentiment = None

        self.model_class1 = None
        self.model_class2 = None
        self.model_class3 = None
        self.model_class4 = None
        self.model_class5 = None
        self.model_class6 = None
        self.model_class7 = None
        self.model_class8 = None
        self.model_class9 = None
        self.model_class10 = None
        self.model_sentiment1 = None
        self.model_sentiment2 = None
        self.model_sentiment3 = None
        self.model_sentiment4 = None
        self.model_sentiment5 = None

        self.ID2label_class = None
        self.ID2label_sentiment = None

        #SQL
        self.conn = None
        self.cur = None

    #SQL
    def sqlConnect(self):
        server = self.config["SQLDB"]["SERVER"]
        database = self.config["SQLDB"]["DB"]
        username = self.config["SQLDB"]["USERNAME"]
        password = self.config["SQLDB"]["PWD"]
        driver= '{ODBC Driver 13 for SQL Server}'
        try:
            self.conn = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)
            self.cur = self.conn.cursor()
        except Exception, e:
            print e.message

    def sqlDisconnect(self):
        try:
            if self.conn is not None:
                self.cur.close()
                self.conn.close()
        except Exception, e:
            print e.message

    def insert_script(self, str_insert, times=1):
        for i in range(times):
            self.cur.execute(str_insert)
            
    # AI Engine
    def sentimentInital(self):
        # AI Engine
        load_start = time.clock()
        self.dict_class = self.load_dict('./code/class_model/class2_weight_dropout0.2.dict')
        self.dict_sentiment = self.load_dict('./code/sentiment_model/sentiment_weight_no_reduce_produce.dict')

        self.model_class1 = load_model('./code/class_model/model_class2_dropout0.2_1.h5')
        self.model_class2 = load_model('./code/class_model/model_class2_dropout0.2_2.h5')
        self.model_class3 = load_model('./code/class_model/model_class2_dropout0.2_3.h5')
        self.model_class4 = load_model('./code/class_model/model_class2_dropout0.2_4.h5')
        self.model_class5 = load_model('./code/class_model/model_class2_dropout0.2_5.h5')
        self.model_class6 = load_model('./code/class_model/model_class2_weight_new6.h5')
        self.model_class7 = load_model('./code/class_model/model_class2_weight_new7.h5')
        self.model_class8 = load_model('./code/class_model/model_class2_weight_new8.h5')
        self.model_class9 = load_model('./code/class_model/model_class2_weight_new9.h5')
        self.model_class10 = load_model('./code/class_model/model_class2_weight_new10.h5')
        self.model_sentiment1 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce1.h5')
        self.model_sentiment2 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce2.h5')
        self.model_sentiment3 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce3.h5')
        self.model_sentiment4 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce4.h5')
        self.model_sentiment5 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce5.h5')
        print("load:"+str(time.clock() - load_start))

        self.ID2label_class, self.ID2label_sentiment = self.loadClassSentiment()

    def loadClassSentiment(self):
        classname2ID = {}
        ID2label_class = {}
        file = codecs.open('./data/class2_labels.txt', "r", encoding='utf-8', errors='ignore')
        count = 0
        for line in file.readlines():
            classname2ID[line.strip()] = count
            ID2label_class[count] = line.strip()
            count += 1
        ID2label_senti = {}
        ID2label_senti[0] = u'中性'
        ID2label_senti[1] = u'正向'
        ID2label_senti[2] = u'负向'
        return ID2label_class, ID2label_senti

    def load_dict(self, path):
        file = codecs.open(path, "r", encoding='utf-8',errors='ignore')
        dict = {}
        for line in file.readlines():
            line = line.strip()
            dict[line] = len(dict)
        return dict
        
    def probs2label(self, pred):
        labels = []
        for probs in pred:
            probs = np.asarray(probs)
            index = np.argmax(probs)
            labels.append(index)
        return labels

    def load_class2_to_class1(self):
        file = codecs.open('./data/class2_class1.txt', "r", encoding='utf-8', errors='ignore')
        class2_class1 = {}
        for line in file.readlines():
            line = line.strip()
            terms = line.split('\t')
            class2_class1[terms[0]] = terms[1]
        return class2_class1

    '''
    def predict(self, text, dict, model, ID2label):
        seq = self.text_to_sequence(text, dict)
        sequences = []
        sequences.append(seq)
        X = pad_sequences(sequences, maxlen=100)
        preds = model.predict(X, batch_size=10000, verbose=0)
        labels_pred = self.probs2label(preds)
        label = ID2label[labels_pred[0]]
        return label
    '''

    def predict(self, texts, dict, model, ID2label):
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
            seq = self.text_to_sequence(sen, dict)
            sequences.append(seq)
        X = pad_sequences(sequences, maxlen=100)
        preds = model.predict(X, batch_size=500, verbose=0)
        # print("preds:", preds)
        # print("pl:", len(preds))
        labels_pred = self.probs2label(preds)
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

    def text_to_sequence(self, text, dict):
        seq = []
        for char in text:
            if char in dict:
                seq.append(dict[char])
            else:
                seq.append(dict['UNK'])
        return seq

    def batch_predict(self, text, id):
        text4analysis, idList = self.duplicateCheck(id, text)

        samples = []
        for sent in text4analysis:
            sent_list=re.split(u'[，,。！!？?:：；;]', sent)
            while "" in sent_list:
                sent_list.remove("")
            samples.append(sent_list)
        result = []
        start = time.clock()
        predict_label_class2_1 = self.predict(samples, self.dict_class, self.model_class1, self.ID2label_class)
        predict_label_class2_2 = self.predict(samples, self.dict_class, self.model_class2, self.ID2label_class)
        predict_label_class2_3 = self.predict(samples, self.dict_class, self.model_class3, self.ID2label_class)
        predict_label_class2_4 = self.predict(samples, self.dict_class, self.model_class4, self.ID2label_class)
        predict_label_class2_5 = self.predict(samples, self.dict_class, self.model_class5, self.ID2label_class)
        predict_label_class2_6 = self.predict(samples, self.dict_class, self.model_class6, self.ID2label_class)
        predict_label_class2_7 = self.predict(samples, self.dict_class, self.model_class7, self.ID2label_class)
        predict_label_class2_8 = self.predict(samples, self.dict_class, self.model_class8, self.ID2label_class)
        predict_label_class2_9 = self.predict(samples, self.dict_class, self.model_class9, self.ID2label_class)
        predict_label_class2_10 = self.predict(samples, self.dict_class, self.model_class10, self.ID2label_class)
        predict_label_sentiment1 = self.predict(samples, self.dict_sentiment, self.model_sentiment1, self.ID2label_sentiment)
        predict_label_sentiment2 = self.predict(samples, self.dict_sentiment, self.model_sentiment2, self.ID2label_sentiment)
        predict_label_sentiment3 = self.predict(samples, self.dict_sentiment, self.model_sentiment3, self.ID2label_sentiment)
        predict_label_sentiment4 = self.predict(samples, self.dict_sentiment, self.model_sentiment4, self.ID2label_sentiment)
        predict_label_sentiment5 = self.predict(samples, self.dict_sentiment, self.model_sentiment5, self.ID2label_sentiment)
        class2_class1 = self.load_class2_to_class1()
        predict_label_class1 = []
        predict_label_class2 = []
        predict_label_sentiment = []

        for long in range(len(predict_label_class2_1)):
            #inner_term_class1 = []
            #inner_term_class2 = []
            #inner_term_senti = []

            #selected_sent = []
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
                    #selected_sent.append(samples[long][short])
                    #sent_result.append(str((predict_label_class1_term, predict_label_class2_term, predict_label_sentiment_term)))
                    sub_result = {}
                    sub_result['sentence'] = samples[long][short]
                    sub_result['class1'] = predict_label_class1_term
                    sub_result['class2'] = predict_label_class2_term
                    sub_result['sentiment'] = predict_label_sentiment_term
                    sent_result.append(sub_result)
            
            timestamp=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
            for i in idList[long]:
                for item in sent_result:
                    tmp_list = [i]
                    tmp_list.append(item['class1'])
                    tmp_list.append(item['class2'])
                    tmp_list.append(item['sentence'])
                    tmp_list.append(item['sentiment'])
                    tmp_list.append(timestamp)
                    result.append(tmp_list)

        print("predict:"+str(time.clock() - start))
        return result

    def readConfiguration(self):
        config = configparser.ConfigParser()
        config.read('Configuration.ini')
        return config        

    # Read Data From AdMaster
    def getHeaders(self, method, uri):
        # Get timestamp
        tz = pytz.timezone('Asia/Shanghai')
        timestamp=datetime.datetime.now(tz).isoformat()

        requestInfo = "\n".join([
            method.upper(),
            uri,
            self.config["ADMASTER"]["APP_KEY"],
            timestamp,
            self.config["ADMASTER"]["SIGN_METHOD"],
            self.config["ADMASTER"]["SIGN_VERSION"],
            ""]
        )

        #print "[INFO] requestInfo = " + requestInfo

        signature = base64.b64encode(
            hmac.new(
                self.config["ADMASTER"]["APP_SECRET"].encode('utf-8'), 
                msg=requestInfo.encode('utf-8'), 
                digestmod=hashlib.sha256
            ).digest()
        )

        #print "[INFO} signature = " + str(signature)
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Cache-Control': 'no-cache',
            'X-Auth-Signature': str(signature),
            'X-Auth-Key': self.config["ADMASTER"]["APP_KEY"],
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Sign-Method': self.config["ADMASTER"]["SIGN_METHOD"],
            'X-Auth-Sign-Version': self.config["ADMASTER"]["SIGN_VERSION"]
        }

        return headers

    def getRawDataByRuleIds(self, startDate, endDate, maxResults = 10):
        print("[INFO] get raw data by rule ids...")
        finalRes = []
        comments = []
        for id in self.RuleIds:
            res, com = self.getRawData(startDate, endDate, id, maxResults)
            finalRes += res
            comments += com
        return finalRes, comments

    def getRawData(self, startDate, endDate, ruleId, maxResults = 10):
        finalRawData = []
        comments = []
        #print "[INFO] the url is %s" % url
        params = {}
        params["platforms"] = self.Platforms
        params["metrics"] = ["volume"]
        #params["endDate"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params["endDate"] = endDate
        params["dimensions"] = []
        #params["startDate"] = (datetime.datetime.now()-datetime.timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        params["startDate"] = startDate
        params["action"] = "query"
        params["ruleId"] = ruleId
        params["format"] = "array"
        params["filters"] = []
        params["maxResults"] = maxResults

        j = 0
        done = True
        while (done):
            params["startIndex"] = j*maxResults
            rawData = requests.post(self.url, headers = self.getHeaders("POST", self.URI), data = json.dumps(params))
            try:
                if (rawData.status_code != 200):
                    done = False
                else:
                    if len(rawData.json())<maxResults:
                        done = False
                    # get headers
                    headerCol = rawData.headers._store['x-content-header-cols'][1].replace("%5B","").replace("%22","").replace("%5D","").split(",")
                    for i in range(len(rawData.json())):
                        info = []
                        data = rawData.json()[i]
                        for k in range(len(self.Cols)):
                            if self.Cols[k] in headerCol:
                                index = headerCol.index(self.Cols[k])
                                if type(data[index]) == list:
                                    data[index] = str(data[index])
                                if type(data[index]) == str:
                                    data[index] = data[index].encode('utf-8')
                                info.append(data[index])
                            else:
                                info.append("N/A")

                        comments.append(info[self.Cols.index("content")])
                        #self.writeDB(cnxn, cursor, info, mscomment)
                        finalRawData.append(info)
                        #print "[INFO] the content %s is %s" % (j*maxResults + i, rawData.json()[i])
            except Exception, e:
                print(e.message)
            finally:
                j += 1
        
        #cnxn.close()
        return finalRawData, comments 

    def getQuery(self, params):
        query = ""
        for k in params:
            v = params.get(k)
            if query:
                query = query + "&"
            query = query + str(k) + "=" + str(v)
        query = query.replace("\\+", "%20")
        return query

    # SQL
    def rawDataCMD(self):
        colList = []
        valueList = []
        for i in range(len(self.Cols)):
            colList.append("_"+self.Cols[i])
            valueList.append("?")
        
        sqlCMD = (
            "INSERT INTO admaster (" 
            + ",".join(colList)
            + ") VALUES ("
            + ",".join(valueList)
            + ")"
        )

        return sqlCMD
    
    def sentiDataCMD(self):
        sqlCMD = (
            "INSERT INTO aiengine (rawId, class1, class2, sentence, sentiment, createAt) VALUES (?,?,?,?,?,?)"
        )
        return sqlCMD

    def duplicateCheck(self, id, text):
        result = {}
        for i in range(len(text)):
            if text[i] not in result.keys():
                result[text[i]] = []           
            result[text[i]].append(id[i])

        return result.keys(), result.values()

def datetimeToStr(dt):
    strDt = str(dt.year) + "-" + str(dt.month) + "-" \
        + str(dt.day) + " " + str(dt.hour) + ":" + str(dt.minute) + ":" + str(dt.second)
    return strDt

if __name__=="__main__":
    #print("Get Data from ADMASTER API:")
    dataKit = DataKit()
    dataKit.sentimentInital()
    #finalRes, comments = dataKit.getRawData("2018-01-01 00:00:00", "2018-01-01 00:30:00", 27684)
    start = datetime.datetime(2018, 3, 18, 18, 0, 0)
    delta = datetime.timedelta(hours=1)
    end = start+delta    
    now = datetime.datetime.now()

    while (now > start):
        print(start)        
        finalRes, comments = dataKit.getRawDataByRuleIds(datetimeToStr(start), datetimeToStr(end), 10000)
        numberOfdata = len(finalRes)
        try:
            id = []
            # write rawdata to table
            cmd = dataKit.rawDataCMD()   
            dataKit.sqlConnect()
            dataKit.cur.executemany(cmd, finalRes)
            dataKit.conn.commit()
            dataKit.cur.execute("SELECT top %s id from admaster order by id desc" % numberOfdata)
            row = dataKit.cur.fetchall()
            dataKit.sqlDisconnect()
            row = [row[i][0] for i in range(numberOfdata)]
            id = row[::-1]
            if (len(comments) == len(id)):
                sentiments = dataKit.batch_predict(comments, id)
                # write sentiments to table sentiment
                cmd = dataKit.sentiDataCMD()
                dataKit.sqlConnect()
                dataKit.cur.executemany(cmd, sentiments)
                dataKit.conn.commit()
                dataKit.sqlDisconnect()
        except Exception,e:
            print(e.message)
        finally:
            start = end
            end = start + delta
            now = datetime.datetime.now()
