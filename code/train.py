from DataGenerator import DataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist, cifar10
from keras import utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import nets as nets

num_classes = 10
epochs = 100
batch_size = 250

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)
# x_train = x_train.resize(60000, 28, 28, 1)
# x_test = x_test.resize(10000, 28, 28, 1)
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# # Parameters
# params = {'batch_size': batch_size,
#           'n_classes': num_classes,
#           'n_channels': x_train.shape[-1],
#           'shuffle': True}

# # Generators
# training_generator = DataGenerator(x_train, y_train, **params)
# validation_generator = DataGenerator(x_test, y_test, **params)

shift = 0.3
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.5,
    # zoom_range=0.2,
    # rotation_range=90,
    width_shift_range=shift, height_shift_range=shift,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # zca_whitening=True,
    horizontal_flip=True,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_datagen.fit(x_train)
test_datagen.fit(x_test)

model = nets.Net_v3()
model.build(input_shape=(None,32,32,3))
model.compile(loss="categorical_crossentropy",
              optimizer="sgd", metrics=['accuracy'])
model.summary()

filepath = '/Users/precious/defect_detecting/models/classifier'
if not os.path.exists(filepath):
    os.makedirs(filepath)
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

history = model.fit(
                    x=train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    # x=training_generator,
                    validation_data=test_datagen.flow(x_test, y_test, batch_size=batch_size),
                    # validation_data=validation_generator,
                    epochs=epochs,
                    # steps_per_epoch=len(x_train) / batch_size,
                    #   use_multiprocessing=True,
                    #   workers=8,
                    callbacks=callback_list)

# 借助 history 对象了解训练过程
history_dict = history.history
acc = history_dict['accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']

# 借助 Matplotlib 绘制图像
plt.plot(range(1, len(acc)+1), acc, 'b--')
plt.plot(range(1, len(loss)+1), loss, 'r-')
# plt.plot(range(1, len(val_loss)+1), val_loss, '-')
plt.plot(range(1, len(val_accuracy)+1), val_accuracy, '--')

# 显示图例
plt.legend(['accuracy', 'loss', 'val_accuracy'])
plt.show()
