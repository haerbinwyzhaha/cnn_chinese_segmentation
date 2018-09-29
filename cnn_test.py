import numpy as np
import tensorflow as tf
import cnn_train
import json


def get_test_data(pure_txts,pure_tags,word_id,tag2vec):
    x=[]
    y=[]
    for i in range(len(pure_txts)):
        x.extend([word_id.get(j,4726) for j in pure_txts[i]])
        y.extend([tag2vec[j] for j in pure_tags[i]])
    return [x],[y]

def model_test(vac_size,x_data,y_data,predict=False):
    embedding_size=128
    keep_prob=tf.placeholder(tf.float32)
    embeddings=tf.Variable(tf.random_uniform([vac_size,embedding_size],-1,1))

    x=tf.placeholder(tf.int32,shape=[None,None])
    embedded=tf.nn.embedding_lookup(embeddings,x)
    embedded_dropout=tf.nn.dropout(embedded,keep_prob)

    W1=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size],-1,1))
    b1=tf.Variable(tf.random_uniform([embedding_size],-1,1))
    a1=tf.nn.relu(tf.nn.conv1d(embedded_dropout,W1,stride=1,padding='SAME')+b1)

    W2=tf.Variable(tf.random_uniform([3,embedding_size,int(embedding_size/4)],-1,1))
    b2=tf.Variable(tf.random_uniform([int(embedding_size/4)],-1,1))
    a2=tf.nn.relu(tf.nn.conv1d(a1,W2,stride=1,padding='SAME')+b2)

    W3=tf.Variable(tf.random_uniform([3,int(embedding_size/4),4],-1,1))
    b3=tf.Variable(tf.random_uniform([4],-1,1))
    y_pre=tf.nn.softmax(tf.nn.conv1d(a2,W3,stride=1,padding='SAME')+b3)

    y=tf.placeholder(tf.float32,shape=[None,None,4])

    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,'./model_data/model_data/frist_model.ckpt')

    # print(sess.run(W3),W3.shape)
    correct_pre=tf.equal(tf.argmax(y,2),tf.argmax(y_pre,2))
    acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))

    if predict:
        result = sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        return result

    else:
        sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        scores = sess.run(acc, feed_dict={x: x_data, y: y_data, keep_prob: 1.0})
        print(scores)

def viterbi(result,trans_pro):
    nodes=[dict(zip( ('S','B','M','E'),i )) for i in result]
    paths=nodes[0]
    for t in range(1,len(nodes)):
        path_old=paths.copy()
        paths={}
        for i in nodes[t]:
            nows={}
            for j in path_old:
                if j[-1]+i in trans_pro:
                    nows[j+i]=path_old[j]+nodes[t][i]+trans_pro[j[-1]+i]
            pro,key=max([(nows[key],key) for key,value in nows.items()])
            paths[key]=pro
    best_pro,best_path=max([(paths[key],key)for key,value in paths.items()])
    return best_path

def segword(txt,best_path):
    begin,end=0,0
    seg_word=[]
    for index,char in enumerate(txt):
        signal=best_path[index]
        if signal=='B':
            begin=index
        elif signal=='E':
            seg_word.append(txt[begin:index+1])
            end=index+1
        elif signal=='S':
            seg_word.append(char)
            end=index+1
    if end<len(txt):
        seg_word.append(txt[end:])
    return seg_word

def cnn_seg(txt):
    word_id=json.load(open('vacabulary.json','r'))
    vacabulary_size=len(word_id)+1
    trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}
    trans_pro={state:np.log(num) for state,num in trans_pro.items()}

    txt2id=[[word_id.get(word,4726)for word in txt]]
    result=model_test(vacabulary_size,x_data=txt2id,y_data=None,predict=True)
    result = result[0, :, :]
    best_path=viterbi(result,trans_pro)

    return  segword(txt,best_path)

def cnn_test(path="./corpus_data/msr_test_gold.utf8"):
    pure_txts, pure_tags = cnn_train.get_corpus(path)
    word_id=json.load(open('vacabulary.json','r'))
    vacabulary_size=len(word_id)+1
    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    x, y = get_test_data(pure_txts, pure_tags, word_id, tag2vec)
    model_test(vacabulary_size, x_data=x, y_data=y)

if __name__ == '__main__':
    # print(cnn_test())

    ##测试句子分词效果
    print(cnn_seg("我爱中国"))

    ## 模型在测试集上的效果
    # print(cnn_test())
    
    #-----以下为各种函数的编写过程---------------
    #获取测试集的数据
    # pure_txts,pure_tags=cnn_crf.get_corpus("./corpus_data/msr_test_gold.utf8")
    # tf_config=cnn_crf.make_default()
    # word_id=json.load(open('vacabulary.json','r'))
    # vacabulary_size=len(word_id)+1
    # print(vacabulary_size)
    # tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    # x, y = get_test_data(pure_txts, pure_tags, word_id, tag2vec)
    # # model_test(vacabulary_size,x_data=x,y_data=y)
    #
    # trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}
    # trans_pro={state:np.log(num) for state,num in trans_pro.items()}
    # print(trans_pro)

    # txt="我爱中国"
    # test_txt="我爱中国"
    # test_txt=[[word_id.get(word,4726)for word in test_txt]]
    # result=model_test(vacabulary_size,x_data=test_txt,y_data=None,predict=True)
    # result=result[0,:,:]

    # nodes=np.random.random((6,4))
    # print(nodes)
    # nodes=[dict(zip(('S','B','M','E') , i)) for i in result]
    # paths=nodes[0]
    # # print(trans_pro['ss'])
    # for t in range(1,len(nodes)):
    #     path_old=paths.copy()
    #     paths={}
    #     for i in nodes[t]:
    #         nows={}
    #         for j in path_old:
    #             if j[-1]+i in trans_pro:
    #                 nows[j+i]=path_old[j]+nodes[t][i]+trans_pro[j[-1]+i]
    #         pro,key=max([(nows[key],key) for key,value in nows.items()])
    #         # print(nows,pro,key)
    #         paths[key]=pro
    # best_pro,best_path=max([(paths[key],key) for key,value in paths.items()])
    # print(best_path)
    # best_path=viterbi(result)
    # print(segword(txt,best_path))

    # seg_word=[]
    # start,end=0,0
    # for index,char in enumerate(txt):
    #     signal=best_path[index]
    #     if signal=='B':
    #         begin=index
    #     elif signal=='E':
    #         seg_word.append(txt[begin:index+1])
    #         end=index+1
    #     elif signal=='S':
    #         seg_word.append(char)
    #         end=index+1
    # if end<len(txt):
    #     seg_word.append(txt[end:])
    # print(seg_word)



    # Viterbi version-1
    #S-0,B-1,M-2,E-3
    #第一个节点初始化
    # paths=nodes[0]
    # print(paths)

    # test_txt="虽然一路上队伍里肃静无声"
    # test_txt=[[word_id.get(word,4726)for word in test_txt]]
    # result=model_test(vacabulary_size,x_data=test_txt,y_data=None,predict=True)
    # result=result[0,:,:]
    # print(result.shape,'\n',result)
    # nodes=result
    # path_v=[{}]
    # path_max={}
    # for state,i in zip(states,range(len(states))):
    #     path_v[0][state]=nodes[0][i]
    #     path_max[state]=[state]
    # for t in range(1,len(nodes)):
    #     path_v.append({})
    #     new_path={}
    #     for state in states:
    #         (temp_pro,temp_state)=max([(path_v[t-1][y]+nodes[t][k],y) for y,k in zip(states,range(len(states)))])
    #         path_v[t][state]=temp_pro
    #         new_path[state]=path_max[temp_state]+[state]
    #     path_max=new_path
    # best_path_pro,last_state=max([(path_v[len(nodes)-1][y0],y0) for y0,j in zip(states,range(len(states)))])
    # print(path_max[last_state])
