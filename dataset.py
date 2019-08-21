import os
import numpy as np
import re
from collections import OrderedDict #维护字典有序插入
from keras.utils import np_utils

tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}

#判断是否为中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

#统计词频
def word_count(data):
    #data tyep:list
    word_freq={}
    for sen in data:
        for word in sen.split():
            if is_Chinese(word):
                for char in word:
                    word_freq[char]=word_freq.get(char,0)+1
            else:
                word_freq[word]=word_freq.get(word,0)+1
    return word_freq

#按照词频顺序逆序排序
def word_index(word_freq):
    #word_freq type:dict
    stat=sorted(word_freq.items(),key=lambda x:x[1],reverse=True)
    words=[s[0] for s in stat]
    char2id={c:i+1 for i,c in enumerate(words)}
    id2char={i+1:c for i,c in enumerate(words)}
    # 将字典char2id写入文件
    fw = open("dictionary/char2id.txt", 'w', encoding='utf-8')
    fw.write(str(char2id))  # 把字典转化为str
    fw.close()
    #将字典id2char写入文件
    f=open("dictionary/id2char.txt",'w',encoding='utf-8')
    f.write(str(id2char))
    f.close()

    return char2id, id2char

#按符号切分数据集
def data_cut(vocab):
    sentences=[]
    for sen in vocab:
        sen=re.split('[。\n]', sen)
        for s in sen:
            if len(s)>0:
                sentences.append(s)
    return sentences

def data_file_cut(filename):
    data=open(filename,encoding='utf-8').read().rstrip()
    data = re.split('[。\n]', data)
    #print(len(data))
    sentences = []
    for i in data:
        if len(i) > 0:
            sentences.append(i)
    return sentences

#将数据集转换成向量
def load_data(sentences,char2id):
    #准备数据
    maxlen=250
    X_data=[]
    y_data=[]
    dic = OrderedDict()
    for sentence in sentences:
        sentence=sentence.split()
        X=[]
        y=[]

        for word in sentence:
            word=word.strip()
          #print(word)
            if is_Chinese(word):
                if len(word)==0:
                    continue
                elif len(word)==1 and word in char2id:
                    X.append(char2id[word])
                    y.append(tags['s'])
                    #dic[word]='s'
                elif len(word)>1:
                    X.append(char2id[word[0]])
                    y.append(tags['b'])
                    #dic[word[0]]='b'
                    for i in range(1,len(word)-1):
                        X.append(char2id[word[i]])
                        y.append(tags['m'])
                        #dic[word[i]]='m'
                    X.append(char2id[word[-1]])
                    y.append(tags['e'])
                    #dic[word[-1]]='e'
            else:
                X.append(char2id[word])
                y.append(tags['s'])
                #dic[word]='s'

        # 统一长度    一个小句子的长度不能超过46,否则将其切断。只保留46个
        if len(X) > maxlen:
            X = X[:maxlen]
            y = y[:maxlen]
        else:
            for i in range(maxlen - len(X)):  # 如果长度不够的，我们进行填充，记得标记为x
                X.append(0)
                y.append(tags['x'])

        if len(X)>0:
            X_data.append(X)
            y_data.append(y)
    X=np.array(X_data)
    y=np_utils.to_categorical(y_data,5)
    #将字典写入文件
    # fw = open("train_label_dic.txt", 'w', encoding='utf-8')
    # fw.write(str(dic))  # 把字典转化为str
    # fw.close()

    return X,y

if __name__ == '__main__':
    vocab=[]
    train_path='data/train_cws.txt'
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line != None:
                vocab.append(line.rstrip('\n'))
    word_freq=word_count(vocab) #词频字典
    char2id, id2char=word_index(word_freq) #索引字典
    sentences=data_cut(vocab)
    X_train, y_train = load_data(sentences,char2id)
    print(X_train)
    print(y_train)


