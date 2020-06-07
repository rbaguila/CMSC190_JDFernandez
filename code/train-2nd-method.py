import warnings
warnings.filterwarnings('ignore') # suppress import warnings

import os
import cv2
import tflearn
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

''' <global actions> '''

TRAIN_DIR = 'train/80'
# TRAIN_DIR = 'train/test'

TEST_DIR = 'test/test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'jycropdisease-{}-{}.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
tf.reset_default_graph()

''' </global actions> '''

def label_leaves(leaf):

    leaftype = leaf
    # print(leaf)
    ans = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if leaftype == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ans =[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Corn_(maize)___Common_rust_': ans =[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Corn_(maize)___healthy': ans =[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Corn_(maize)___Northern_Leaf_Blight': ans =[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Pepper,_bell___Bacterial_spot': ans =[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Pepper,_bell___healthy': ans =[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Potato___Early_blight': ans =[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Potato___healthy': ans =[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Potato___Late_blight': ans =[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Tomato___Bacterial_spot': ans =[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif leaftype == 'Tomato___Early_blight': ans =[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    elif leaftype == 'Tomato___healthy': ans =[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    elif leaftype == 'Tomato___Late_blight': ans =[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    elif leaftype == 'Tomato___Leaf_Mold': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    elif leaftype == 'Tomato___Septoria_leaf_spot': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif leaftype == 'Tomato___Spider_mites Two-spotted_spider_mite': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    elif leaftype == 'Tomato___Target_Spot': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    elif leaftype == 'Tomato___Tomato_mosaic_virus': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    elif leaftype == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ans =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

 
    return ans

def create_training_data():

    training_data = []
    indeces =[]
    typeNumber=0
    totalData=0
    for folder in os.listdir(TRAIN_DIR):
        indeces.append([totalData,totalData])
        for img in  tqdm(os.listdir(TRAIN_DIR+"/"+folder)):

            label = label_leaves(folder)
            path = os.path.join(TRAIN_DIR+"/"+folder,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img),np.array(label)])
            indeces[typeNumber][1] = totalData
            totalData+=1
        typeNumber+=1
    np.save('train_data.npy', training_data)
    print(indeces)
    return [training_data,indeces]

def split_training_data(train_data,data_indeces,k_fold,fold_number):
    train =np.array([], dtype=np.uint8).reshape(0,2)
    test =np.array([], dtype=np.uint8).reshape(0,2)
    for index in data_indeces:
        data = train_data[index[0]:index[1]+1]

        splitted_data = np.array_split(data,k_fold)

        new_test_data = splitted_data[fold_number]
       
        new_train_data = np.concatenate((np.delete(splitted_data, fold_number, 0)))
    

        train = np.concatenate((train,new_train_data))
        test = np.concatenate((test, new_test_data))
    return{'train_data':train, 'test_data':test}

def main():
    f = open("results.txt", "w+")

    get_data = create_training_data()
    train_data = get_data[0]
    data_indeces =get_data[1]


    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 19, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')


    k_fold =5
    fold_number=0

    MODEL_NEW_NAME = 'jycropdisease-fold-new{}-{}-{}.model'.format(fold_number+1,LR, '2conv-basic')

    data=train_data
    print("======= FOLD %d =======" % (fold_number+1))
    split_data = split_training_data(data,data_indeces,k_fold,fold_number)

    train = split_data['train_data']
    test = split_data['test_data']


    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=1000, show_metric=True, run_id=MODEL_NEW_NAME)
    

    model.save(MODEL_NEW_NAME)
    f.close()

if __name__ == '__main__': main()
