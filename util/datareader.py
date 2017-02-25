import csv
import numpy as np
import model.docModel as docModel

CLASS_NUM=10

def classify_part_02(param):
#classfity the part param range from 0 to 2
    return round((float(param)/2)*CLASS_NUM)

def classify_part_01(param):
#classfity the part param range from 0 to 1
    return round((float(param))*CLASS_NUM)

def classify_part_11(param):
#classfity the part param range from -1 to 1
    return round(((float(param)+float(1))/2)*CLASS_NUM)

def classify_part_10(param):
#classfity the part param range from -1 to 0
    return round(((float(param)+float(1)))*CLASS_NUM)

def getClassParam_02(param):
#convert the class into a part param range from 0 to 2
    return (float(param)/CLASS_NUM)*2

def getClassParam_01(param):
#convert the class into a part param range from 0 to 1
    return (float(param)/CLASS_NUM)

def getClassParam_11(param):
#convert the class into a part param range from -1 to 1
    return ((float(param)/CLASS_NUM)*2-float(1))

def getClassParam_10(param):
#convert the class into a part param range from -1 to 0
    return ((float(param)/CLASS_NUM)-float(1))

def getClassVector(raw_y):
#classify the face according to all of the part classes into one-hot vector
    y=np.zeros([CLASS_NUM**19], dtype=np.int32)
    y[raw_y]=1
    return y

def read_raw_data1(path, sen_vec_size, face_vec_size):
    #into CLASS_NUM**19 one hot
    raw_data=[]
    f=open(path,"rb")
    for row in csv.reader(f):
        raw_data.append(row)
    data_len=len(raw_data)
    raw_data_x=np.zeros([data_len, sen_vec_size], dtype=np.float32)
    raw_data_y=np.zeros([data_len], dtype=np.int32)
    for i in range(data_len):
        raw_data_x[i]=np.array(docModel.getVector(raw_data[i][0]), dtype=np.float32)
        raw_data_y[i]=(classify_part_02(raw_data[i][1])+1) + \
                      classify_part_01(raw_data[i][2]) * CLASS_NUM**1 + \
                      classify_part_02(raw_data[i][3]) * CLASS_NUM**2 + \
                      classify_part_01(raw_data[i][4]) * CLASS_NUM**3 + \
                      classify_part_11(raw_data[i][5]) * CLASS_NUM**4 + \
                      classify_part_11(raw_data[i][6]) * CLASS_NUM**5 + \
                      classify_part_11(raw_data[i][7]) * CLASS_NUM**6 + \
                      classify_part_10(raw_data[i][8]) * CLASS_NUM**7 + \
                      classify_part_11(raw_data[i][9]) * CLASS_NUM**8 + \
                      classify_part_11(raw_data[i][10]) *CLASS_NUM**9 + \
                      classify_part_11(raw_data[i][11]) *CLASS_NUM**10 + \
                      classify_part_11(raw_data[i][12]) *CLASS_NUM**11 + \
                      classify_part_11(raw_data[i][13]) *CLASS_NUM**12 + \
                      classify_part_11(raw_data[i][14]) *CLASS_NUM**13 + \
                      classify_part_11(raw_data[i][15]) *CLASS_NUM**14 + \
                      classify_part_11(raw_data[i][16]) *CLASS_NUM**15 + \
                      classify_part_11(raw_data[i][17]) *CLASS_NUM**16 + \
                      classify_part_01(raw_data[i][18]) *CLASS_NUM**17 + \
                      classify_part_01(raw_data[i][19]) *CLASS_NUM**18
    return raw_data_x, raw_data_y

def read_raw_data2(path, sen_vec_size, face_vec_size):
#into class vector of each part. for lable part respectively
    raw_data=[]
    f=open(path,"rb")
    for row in csv.reader(f):
        raw_data.append(row)
    data_len=len(raw_data)
    raw_data_x=np.zeros([data_len, sen_vec_size], dtype=np.float32)
    raw_data_y=np.zeros([data_len, face_vec_size], dtype=np.int32)
    for i in range(data_len):
        raw_data_x[i]=np.array(docModel.getVector(raw_data[i][0]), dtype=np.float32)
        raw_data_y[i][0]=classify_part_02(raw_data[i][1])
        raw_data_y[i][1]=classify_part_01(raw_data[i][2])
        raw_data_y[i][2]=classify_part_02(raw_data[i][3])
        raw_data_y[i][3]=classify_part_01(raw_data[i][4])
        raw_data_y[i][4]=classify_part_11(raw_data[i][5])
        raw_data_y[i][5]=classify_part_11(raw_data[i][6])
        raw_data_y[i][6]=classify_part_11(raw_data[i][7])
        raw_data_y[i][7]=classify_part_10(raw_data[i][8])
        raw_data_y[i][8]=classify_part_11(raw_data[i][9])
        raw_data_y[i][9]=classify_part_11(raw_data[i][10])
        raw_data_y[i][10]=classify_part_11(raw_data[i][11])
        raw_data_y[i][11]=classify_part_11(raw_data[i][12])
        raw_data_y[i][12]=classify_part_11(raw_data[i][13])
        raw_data_y[i][13]=classify_part_11(raw_data[i][14])
        raw_data_y[i][14]=classify_part_11(raw_data[i][15])
        raw_data_y[i][15]=classify_part_11(raw_data[i][16])
        raw_data_y[i][16]=classify_part_11(raw_data[i][17])
        raw_data_y[i][17]=classify_part_01(raw_data[i][18])
        raw_data_y[i][18]=classify_part_01(raw_data[i][19])
    return raw_data_x, raw_data_y

def data_iterator_lstm(path, sen_vec_size, batch_size, steps, part=None):
#part 0~19. if part given, return the class of the part as y
    face_vec_size=19
    num_steps=steps
    raw_data=[]
    f=open(path,"rb")
    for row in csv.reader(f):
        raw_data.append(row)
    f.close()
    data_len=len(raw_data)
    raw_data=None
    batch_len=data_len//batch_size
    data_x=np.zeros([batch_size, batch_len, sen_vec_size], dtype=np.float32)
    data_y=np.zeros([batch_size, batch_len], dtype=np.int32)    
    if part==None:
        raw_data_x, raw_data_y = read_raw_data1(path, sen_vec_size, face_vec_size)
    else:
        raw_data_x, raw_data_y_vec = read_raw_data2(path, sen_vec_size, face_vec_size)
        raw_data_y=np.zeros([data_len], dtype=np.int32)
        for i in range(data_len):
            raw_data_y[i]=raw_data_y_vec[i][part]
    for i in range(batch_size):
        data_x[i] = raw_data_x[batch_len * i:batch_len * (i + 1)]
        data_y[i] = raw_data_y[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data_x[:, i*num_steps:(i+1)*num_steps]
        y = data_y[:, i*num_steps:(i+1)*num_steps]
        yield (x, y)

def data_iterator_softmax(path, sen_vec_size, batch_size, part=None):
#part 0~19. if part given, return the class of the part as y
#TODO:a iterator for the one-softmax-layer nn
#     transform y into size 10 one-hot vector
    face_vec_size=19
    num_steps=steps=1
    raw_data=[]
    f=open(path,"rb")
    for row in csv.reader(f):
        raw_data.append(row)
    f.close()
    data_len=len(raw_data)
    raw_data=None
    batch_len=data_len//batch_size
    data_x=np.zeros([batch_size, batch_len, sen_vec_size], dtype=np.float32)
    data_y=np.zeros([batch_size, batch_len, 11], dtype=np.int32)    
    if part==None:
        raw_data_x, raw_data_y = read_raw_data1(path, sen_vec_size, face_vec_size)
    else:
        raw_data_x, raw_data_y_vec = read_raw_data2(path, sen_vec_size, face_vec_size)
        raw_data_y=np.zeros([data_len, 11], dtype=np.int32)
        for i in range(data_len):
            y_one_hot=np.zeros(11, dtype=np.float32)
            while raw_data_y_vec[i][part]>10:
                raw_data_y_vec[i][part]=raw_data_y_vec[i][part]/2
            y_one_hot[raw_data_y_vec[i][part]]=1
            raw_data_y[i]=y_one_hot
    for i in range(batch_size):
        data_x[i] = raw_data_x[batch_len * i:batch_len * (i + 1)]
        data_y[i] = raw_data_y[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data_x[:, i]
        y = data_y[:, i]
        yield (x, y)
