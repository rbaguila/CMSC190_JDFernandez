
from tqdm import tqdm
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import tensorflow as tf
import tensorflow
import numpy as np
import tflearn
import cv2
import sys
import os
import warnings
warnings.filterwarnings('ignore')  # suppress import warnings


''' <global actions> '''

IMG_SIZE = 50
LR = 1e-3
FOLD = 5
MODEL_NAME = 'model/jycropdisease-fold{}-{}-{}.model'.format(
    FOLD, LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR)  # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow gpu logs

''' </global actions> '''


def process_verify_data(filepath):

    verifying_data = []

    img_name = filepath.split('.')[0]
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    verifying_data = [np.array(img), img_name]

    np.save('verify_data.npy', verifying_data)

    return verifying_data


def analysis(filepath):

    verify_data = process_verify_data(filepath)

    str_label = "Cannot make a prediction."
    status = "Error"

    tf.reset_default_graph()

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
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
    else:
        print('Error: Create a model using neural_network.py first.')

    img_data, img_name = verify_data[0], verify_data[1]

    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
    elif np.argmax(model_out) == 1:
        str_label = 'Corn_(maize)___Common_rust_'
    elif np.argmax(model_out) == 2:
        str_label = 'Corn_(maize)___healthy'
    elif np.argmax(model_out) == 3:
        str_label = 'Corn_(maize)___Northern_Leaf_Blight'
    elif np.argmax(model_out) == 4:
        str_label = 'Pepper,_bell___Bacterial_spot'
    elif np.argmax(model_out) == 5:
        str_label = 'Pepper,_bell___healthy'
    elif np.argmax(model_out) == 6:
        str_label = 'Potato___Early_blight'
    elif np.argmax(model_out) == 7:
        str_label = 'Potato___healthy'
    elif np.argmax(model_out) == 8:
        str_label = 'Potato___Late_blight'
    elif np.argmax(model_out) == 9:
        str_label = 'Tomato___Bacterial_spot'
    elif np.argmax(model_out) == 10:
        str_label = 'Tomato___Early_blight'
    elif np.argmax(model_out) == 11:
        str_label = 'Tomato___healthy'
    elif np.argmax(model_out) == 12:
        str_label = 'Tomato___Late_blight'
    elif np.argmax(model_out) == 13:
        str_label = 'Tomato___Leaf_Mold'
    elif np.argmax(model_out) == 14:
        str_label = 'Tomato___Septoria_leaf_spot'
    elif np.argmax(model_out) == 15:
        str_label = 'Tomato___Spider_mites Two-spotted_spider_mite'
    elif np.argmax(model_out) == 16:
        str_label = 'Tomato___Target_Spot'
    elif np.argmax(model_out) == 17:
        str_label = 'Tomato___Tomato_mosaic_virus'
    elif np.argmax(model_out) == 18:
        str_label = 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'

    label = str_label
    num_label = np.argmax(model_out)

    return label, num_label


def main():
    # filepath = input("Enter Image File Name:\n")
    filepath = "test/dataset(20)"
    for folder in os.listdir(filepath):
        f = open("test/results"+"/"+folder+"-fold5.txt", "w+")
        correct = 0
        total = 0
        prediction = [0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for img in tqdm(os.listdir(filepath+"/"+folder)):

            label, num_label = analysis(filepath+"/"+folder+"/"+img)
            f.write("%s, Expected value: %s, Findings: %s \n" %
                    (img, folder, label))
            total += 1

            if(folder == label):
                correct += 1
            prediction[num_label] += 1

        f.write("Total items: %d" % (total))
        f.write("Correct findings: %d" % (correct))
        f.write("Accuracy: %d" % ((correct/total)*100))
        f.write("\nPredicted Label:")
        for score in prediction:
            f.write("%d " % (score))
        f.close()


if __name__ == '__main__':
    main()
