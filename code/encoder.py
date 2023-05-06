import nets
from keras.datasets import mnist,cifar10
from keras.callbacks import ModelCheckpoint
import os
import keras

name = 'cifar10'

if name == 'mnist':
    input_shape = 28
    channel_shape = 1
elif name =='cifar10':
    input_shape = 32
    channel_shape = 3
    
(x_train, y_train), (x_test, y_test) = eval(name).load_data()

x_train = x_train / 255

model = nets.Main_Net()
model.build(input_shape=(None,input_shape,input_shape,channel_shape))
model.compile(loss="categorical_crossentropy",
              optimizer="sgd")
model.summary()

filepath = './models/encoder'
if not os.path.exists(filepath):
    os.makedirs(filepath)
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='loss', verbose=1, save_best_only=True)
callback_list = [checkpoint]

if name == 'mnist':
    model.fit(x=x_train,y=x_train, epochs=1, batch_size=600, callbacks=callback_list)
elif name == 'cifar10':
    model.fit(x=x_train,y=x_train, epochs=4, batch_size=500, callbacks=callback_list)

from keras import backend as K
from keras.utils import array_to_img

def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]

# input_x = np.array(Image.open('xx.png'))
# input_x = np.expand_dims(input_x, 0)
# img = get_layer_output(model, input_x)

model = keras.models.load_model(filepath)

img = x_train[0].reshape(1,input_shape,input_shape,channel_shape)
img_pred = model.predict(x_train[0].reshape(1,input_shape,input_shape,channel_shape))

img=array_to_img(img[0]*255)
img.save('./pic/{}.png'.format(name))

# print(img, img_pred)

img_pred=array_to_img(img_pred[0]*255)
img_pred.save('./pic/{}_pred.png'.format(name))
