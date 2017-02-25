#coding=utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.doc2vec import *
import MeCab
import sys

if __name__=="__main__":
    model=Doc2Vec.load("bccwj_vec")
    #s="『 無尽蔵 殺人 事件 』 と 『 フォーカス 』"
    #print model.infer_vector(s.split(" "))
    mec=MeCab.Tagger("-Ochasen")
    inp=[]
    while True:
        line=sys.stdin.readline()
        if not line:
            break
        node=mec.parseToNode(line[:-1])
        while node:
            #print node.surface+"\n"
            inp.append(node.surface)
            node=node.next
        print model.infer_vector(inp)
