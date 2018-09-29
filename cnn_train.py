import re
import numpy as np
import json
from collections import Counter,defaultdict
import tqdm
import os
import tensorflow as tf
def make_default():
    #设置在0卡上运行,占用上限50%
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.5

    return  config

def makelabel(word):
    if len(word)==1:
        return 'S'
    else :
        return 'B'+(len(word)-2)*'M'+'E'
def get_corpus(path):
    pure_tags=[]
    pure_txts=[]
    stops=u'，。！？；、：,\.!\?;:\n'
    i=0
    with open(path,'r') as f:
        txt=[line.strip(' ') for line in re.split('['+stops+']',f.read()) if line.strip(' ')]
        for line in txt:
            i+=1
            pure_txts.append('')
            pure_tags.append('')
            for word in re.split(' +',line):
                pure_txts[-1]+=word
                pure_tags[-1]+=makelabel(word)

    ls=[len(i) for i in pure_txts]
    # print(ls)
    ls=np.argsort(ls)[::-1]#从大到小排序
    pure_txts=[pure_txts[i] for i in ls]
    pure_tags=[pure_tags[i] for i in ls]
    return  pure_txts,pure_tags
def data(pure_txts,pure_tags,word_id,tag2vec,batch_size=256):
    # batch_size = 256
    l=len(pure_txts[0])
    x=[]
    y=[]
    for i in range(len(pure_txts)):
        if len(pure_txts[i])!=l or len(x)==batch_size:
            yield x,y
            x=[]
            y=[]
            l=len(pure_txts[i])
        x.append([word_id[j] for j in pure_txts[i]])
        y.append([tag2vec[j] for j in pure_tags[i]])


def cnn_train(pure_txt,pure_tags,word_id,tag2vec,config,epoch=300):
    embedding_size = 128
    keep_prob = tf.placeholder(tf.float32)

    embeddings = tf.Variable(tf.random_uniform([vacabulary_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
    # define x&y
    x = tf.placeholder(tf.int32, shape=[None, None])
    embedded = tf.nn.embedding_lookup(embeddings, x)
    embedded_dropout = tf.nn.dropout(embedded, keep_prob)
    # W1-n0xn0x3
    W1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
    b1 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0), dtype=tf.float32)
    a1 = tf.nn.relu(tf.nn.conv1d(embedded_dropout, W1, stride=1, padding='SAME') + b1)
    # W2=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size/4],-1.0,1.0),dtype=tf.float32)
    W2 = tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size / 4)], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([int(embedding_size / 4)], -1.0, 1.0))
    a2 = tf.nn.relu(tf.nn.conv1d(a1, W2, stride=1, padding='SAME') + b2)
    print(a2.shape)
    W3 = tf.Variable(tf.random_uniform([3, int(embedding_size / 4), 4], -1.0, 1.0))
    print(W3.shape)
    b3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
    a3 = tf.nn.softmax(tf.nn.conv1d(a2, W3, stride=1, padding='SAME') + b3)

    y_ = tf.placeholder(tf.float32, shape=[None, None, 4])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(a3 + 1e-20))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(a3, 2), tf.argmax(y_, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    # epoch = 100

    for i in range(epoch):
        temp_data = tqdm.tqdm(data(pure_txts, pure_tags, word_id, tag2vec,batch_size=512), desc=u'Epcho %s,Accuracy:0.0' % (i + 1))
        k = 0
        accs = []
        for x_data, y_data in temp_data:
            k += 1
            if k % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: x_data, y_: y_data, keep_prob: 1})
                accs.append(acc)
                temp_data.set_description('Epcho %s, Accuracy: %s' % (i + 1, acc))
            sess.run(train_step, feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
        print(u'Epcho %s Mean Accuracy: %s' % (i + 1, np.mean(accs)))


    saver = tf.train.Saver()
    saver.save(sess, './model_data/frist_model.ckpt')


def word2dic(pure_txts,flat=True):
    min_count=2
    word_count=Counter(''.join(pure_txts))
    word_count=Counter({word:index for word,index in word_count.items() if index>=min_count})
    word_id=defaultdict(int)
    id = 0
    for i in  word_count.most_common():
        id+=1
        word_id[i[0]]=id
    vacabulary_size=len(word_id)+1
    if flat:
     json.dump(word_id,open('vacabulary.json','w'))
    return word_count,word_id,vacabulary_size


if __name__ == '__main__':
    pure_txts=[]
    pure_tags=[]
    stops =u'，。！？；、：,\.!\?;:\n'
    txt=None
    with open('corpus_data/msr_training.utf8') as f:
        txt=f.read()
        txt=[word.strip(' ') for word in re.split('['+stops+']',txt) if word.strip(' ')]
        i=0
        for line in txt:
            i+=1
            pure_tags.append('')
            pure_txts.append('')
            for word in re.split(' +',line):
                # print(word)
                pure_txts[-1]+=word
                pure_tags[-1]+=makelabel(word)
            # if(i==20):
            #     break
        # print(pure_txts,'\n',pure_tags)
    ls=[len(i) for i in pure_txts]
    # print(ls)
    ls=np.argsort(ls)[::-1]#从大到小排序
    pure_txts=[pure_txts[i] for i in ls]
    pure_tags=[pure_tags[i] for i in ls]

    word_count,word_id,vacabulary_size=word2dic(pure_txts,flat=False)

    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}

    config=make_default()
    cnn_train(pure_txts,pure_tags,word_id,tag2vec,config,epoch=500)


    ##--------------函数的编写与调试过程----------------------------------------
    # import tensorflow as tf
    # embedding_size=128
    # keep_prob=tf.placeholder(tf.float32)
    #
    # embeddings=tf.Variable(tf.random_uniform([vacabulary_size,embedding_size],-1.0,1.0),dtype=tf.float32)
    # #define x&y
    # x=tf.placeholder(tf.int32,shape=[None,None])
    # embedded=tf.nn.embedding_lookup(embeddings,x)
    # embedded_dropout=tf.nn.dropout(embedded,keep_prob)
    # #W1-n0xn0x3
    # W1=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size],-1.0,1.0),dtype=tf.float32)
    # b1=tf.Variable(tf.random_uniform([embedding_size],-1.0,1.0),dtype=tf.float32)
    # a1=tf.nn.relu(tf.nn.conv1d(embedded_dropout,W1,stride=1,padding='SAME')+b1)
    # # W2=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size/4],-1.0,1.0),dtype=tf.float32)
    # W2= tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size / 4)], -1.0, 1.0))
    # b2=tf.Variable(tf.random_uniform([int(embedding_size/4)],-1.0,1.0))
    # a2=tf.nn.relu(tf.nn.conv1d(a1,W2,stride=1,padding='SAME')+b2)
    # print(a2.shape)
    # W3=tf.Variable(tf.random_uniform([3,int(embedding_size/4),4],-1.0,1.0))
    # print(W3.shape)
    # b3=tf.Variable(tf.random_uniform([4],-1.0,1.0))
    # a3=tf.nn.softmax(tf.nn.conv1d(a2,W3,stride=1,padding='SAME')+b3)
    #
    # y_=tf.placeholder(tf.float32,shape=[None,None,4])
    # cross_entropy=-tf.reduce_sum(y_*tf.log(a3+1e-20))
    # train_step=tf.train.AdamOptimizer().minimize(cross_entropy)
    # correct_prediction=tf.equal(tf.arg_max(a3,2),tf.arg_max(y_,2))
    # accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #
    # init=tf.global_variables_initializer()
    # sess=tf.Session()
    # sess.run(init)
    # epoch=100
    #
    # for i in range(epoch):
    #     temp_data=tqdm.tqdm(data(pure_txts,pure_tags,word_id,tag2vec),desc=u'Epcho %s,Accuracy:0.0'%(i+1))
    #     k=0
    #     accs=[]
    #     for x_data,y_data in temp_data:
    #         k+=1
    #         if k%100==0:
    #             acc=sess.run(accuracy,feed_dict={x:x_data,y_:y_data,keep_prob:1})
    #             accs.append(acc)
    #             temp_data.set_description('Epcho %s, Accuracy: %s' % (i + 1, acc))
    #         sess.run(accuracy,feed_dict={x:x_data,y_:y_data,keep_prob:0.5})
    #     print(u'Epcho %s Mean Accuracy: %s'%(i+1,np.mean(accs)))
    #
    # saver=tf.train.Saver()
    # saver.save(sess,'frist_model.ckpt')
    # print(word_count)
    # print(word_count.most_common())
    # print(np.argsort(ls)[::-1])
    # print(len(txt))
    # stop='，。！？\n'
    # txt=[word.strip( ) for word in re.split('['+stop+']',txt) if word.strip(' ')]
    # print(txt)