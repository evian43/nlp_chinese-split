import re
import numpy as np
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

maxlen=100

#载入字典
with open("dictionary/char2id.txt","r",encoding='utf-8') as f:
    char2id=eval(f.read())

# 定义维特比函数，使用动态规划算法获得最大概率路径
def viterbi(nodes):
    trans = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1] + i in trans.keys():
                    nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
            nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
            paths[nows[0][0]] = nows[0][1]

    paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
    return paths[0][0]


# 使用训练好的模型定义分词函数,用于切分句子
def cut_sentences(data):
    #data = ''.join(data.split())
    data = re.split('([。\n])', data)  # 来一句话，我们先进行切分，因为我们的输入限制在46
    sens = []
    Xs = []
    for sentence in data:
        sen = []
        X = []
        sentence = list(sentence)
        for s in sentence:
            s = s.strip()
            if not s == '' and s in char2id:
                sen.append(s)
                X.append(char2id[s])
        if len(X) > maxlen:
            sen = sen[:maxlen]
            X = X[:maxlen]
        else:
            for i in range(maxlen - len(X)):
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)

    Xs = np.array(Xs)
    ys = model.predict(Xs)  # 对每个字预测出五种概率，其中前四个概率是我们需要的，最后一个概率是对空的预测
    #print(ys)
    results = ''
    for i in range(ys.shape[0]):
        # 将每个概率与sbme对应构成字典 [{s:*, b:*, m:*, e:*}, {}, {}...]
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d[:4])) for d in ys[i]]
        ts = viterbi(nodes)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:
                results += sens[i][x] + ' '
            else:
                results += sens[i][x]

    return results[:-1]

def cut_file(filename):
    with open(filename,'r',encoding='utf-8') as f:
        sentences=f.readlines()
    #data = open(filename, encoding='utf-8').read().rstrip()
        with open('data/test2_seg.txt', 'w', encoding='utf-8') as fw:
            for sentence in sentences:
                sen=cut_sentences(sentence)
                fw.write(sen)
                fw.write('\n')

if __name__=='__main__':
    # 载入模型
    import os
    from keras_contrib.layers import CRF
    crf_layer = CRF(5)
    custom_objects = {'CRF': CRF, 'crf_loss': crf_layer.loss_function, 'crf_viterbi_accuracy': crf_layer.accuracy}
    model = load_model('model/model_bilstm_100_64.hdf5',custom_objects)
    cut_file('test/test2.txt')
