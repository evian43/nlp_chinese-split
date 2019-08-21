from keras.layers import Input,Dense,Embedding,LSTM,Dropout,TimeDistributed,Bidirectional
from keras.models import Model,load_model
from keras.utils import np_utils
from keras.models import Sequential
from collections import OrderedDict
from keras.callbacks import ModelCheckpoint
import numpy as np
import re
import os
from dataset import *
from dataset import data_file_cut

#定义模型所需的参数
embedding_size=100  #字嵌入的长度
maxlen=250   #长于150则截断，短于150则填充0
hidden_size=64
batch_size=64
epochs=10
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}
#载入字典
with open("dictionary/char2id.txt","r",encoding='utf-8') as f:
    char2id=eval(f.read())
    print(char2id)

# #数据集路径
# train_path='data/train_cws.txt'
# test_path='data/test_cws.txt'
# val_path='data/val_cws.txt'
#
#
# #载入数据集
# X_train_sen=data_file_cut(train_path)
# X_val_sen=data_file_cut(val_path)
# X_test_sen=data_file_cut(test_path)
# print(X_test_sen)
#
# #将数据集向量化
# X_train, y_train=load_data(X_train_sen,char2id)
# X_val, y_val=load_data(X_val_sen,char2id)
# X_test, y_test=load_data(X_test_sen,char2id)
#print(X_train)

#定义模型
def model_BILSTM():
    model=Sequential()
    model.add(Embedding(input_dim=len(char2id)+1,output_dim=embedding_size,input_length=maxlen,mask_zero=True))
    model.add(Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='concat'))
    model.add(Dropout(0.6))
    model.add(Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='concat'))
    model.add(Dropout(0.6))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

if __name__=='__main__':
    vocab = []
    train_path = 'data/train_cws.txt'
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line != None:
                vocab.append(line.rstrip('\n'))
    word_freq = word_count(vocab)  # 词频字典
    char2id, id2char = word_index(word_freq)  # 索引字典
    sentences = data_cut(vocab)
    X_train, y_train = load_data(sentences, char2id)

    model=model_BILSTM()
    savePath = "model/model_bilstm.hdf5"         #尽量将模型名字和前面的标题统一，这样便于查找
    checkpoint = ModelCheckpoint(savePath, save_weights_only=False,verbose=1,save_best_only=False, period=1)         #回调函数，实现断点续训功能
    if os.path.exists(savePath):
        model.load_weights(savePath)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    else:
        pass

    #训练模型
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)