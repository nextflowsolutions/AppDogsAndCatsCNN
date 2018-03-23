"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
https://gist.github.com/standarderror/43582e9a15038806da8a846903438ebe
"""

from __future__ import division, print_function, absolute_import
from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # retira a mensagem de alerta AVX do TF pois estou usando somente CPU.

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

###################################
### Importa arquivos de imagem
###################################

files_path = '/train/'

cat_files_path = os.path.join(files_path, 'cat_*.jpg')
dog_files_path = os.path.join(files_path, 'dog_*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
   
###################################
# Preparar amostras de trem e teste
###################################

# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

###################################
# Transformação  das imagens
###################################

# normalização de imagens
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Crie dados de treinamento extra sintéticos girando e girando imagens
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Definir arquitetura de rede
###################################

# Entrada é uma imagem 32x32 com 3 canais de cor (vermelho, verde e azul)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Camada de convolução com 32 filtros, cada um com 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Camada Máx. De Agrupamento
network = max_pool_2d(conv_1, 2)

# 3: Camada de convolução com 64 filtros
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Camada de convolução com 64 filtros
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Camada Máx. De Agrupamento - Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Camada de nó 512 totalmente conectada
network = fully_connected(network, 512, activation='relu')

# 7: Camada de eliminação para combater o overfitting
network = dropout(network, 0.5)

# 8: Camada totalmente conectada com duas saídas
network = fully_connected(network, 2, activation='softmax')

# Configurar como a rede será treinada
acc = Accuracy(name="Precisão")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Envolva a rede em um objeto de modelo
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir=r'C:\Users\rodrigo.fernandes\Documents\Tfs\AppFlowRecognition\App.v1\AppFlowRecognition\Scripts\ClassifierCatsAndDogs\datset\tflearn_logs')

#######################################
# Modelo de treinamento para 100 épocas
#######################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='model_cat_dog_6', show_metric=True)

model.save('model_cat_dog_6_final.tflearn')