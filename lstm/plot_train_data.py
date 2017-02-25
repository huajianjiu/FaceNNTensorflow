import numpy as np
import matplotlib.plot as plt

def data_getter(path, config):
    sen_vec_size=config.sen_vec_size
    face_vec_size=config.face_vec_size
    unroll_step=config.unroll_step

    f=open(path,"rb")
    dataReader=csv.reader(f)
    raw_data=[]
    for row in dataReader:
        raw_data.append(row)
    data_len=len(raw_data)
    raw_data_x=np.zeros([1], dtype=np.float32)
    raw_data_y=np.zeros([data_len, face_vec_size], dtype=np.float32)
    
    for i in range(data_len):
        raw_data_x[i]=np.array(docModel.getVector(raw_data[i][0]), dtype=np.float32)
        #raw_data_y[i]=np.array(raw_data[i][1:20], dtype=np.float32)
#remove the 5-8th, 11th, 12th, 19th parameters because they is almost not used. they will make the loss not fairly computed.
        #raw_data_y[i]=np.array(raw_data[i][1:5]+raw_data[i][9:11]+raw_data[i][13:19], dtype=np.float32)
#use only MouthForm
        raw_data_y[i]=np.array(raw_data[i][17], dtype=np.float32)
        

    x=np.zeros([data_len, unroll_step, sen_vec_size],dtype=np.float32)
    y=np.zeros([data_len, unroll_step, face_vec_size],dtype=np.float32)
    #rnn needs a sequence input and output a sequence
    for i in range(len(raw_data_x)):
        for j in range(unroll_step-1, -1, -1)[:(i+1)]:
            if j-i>=0:
                x[i][j]=raw_data_x[j-i]
                y[i][j]=raw_data_y[j-i]
            else:
                break
    return x,y


