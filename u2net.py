import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
from data import trainGenerator,testGenerator,saveResult

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 50

# train_x_generator = train_datagen.flow_from_directory(
#     './data/intel/train',
#     target_size=(320, 320),
#     batch_size=batch_size,
#     # class_mode='binary'
# )

# train_y_generator = train_datagen.flow_from_directory(
#     './data/intel_target',
#     target_size=(320, 320),
#     batch_size=batch_size,
#     # class_mode='binary'
# )

# validation_generator = test_datagen.flow_from_directory(
#     './data/intel/test',
#     target_size=(320, 320),
#     batch_size=batch_size,
#     # class_mode='binary'
# )

data_gen_args = dict(data_format='channels_first')

train_generator = trainGenerator(batch_size,'./data/intel','train','train_target', data_gen_args, save_to_dir = None, target_size=(320,320))

net = keras.models.load_model("./models/u2net/u2netp_keras.h5")
mse = keras.losses.MeanSquaredError()
net.compile(loss = "mean_squared_error", metrics = ["accuracy"], optimizer="adam")

filepath = '/Users/precious/defect_detecting/models/u2net_keras'
if not os.path.exists(filepath):
    os.makedirs(filepath)
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
callback_list = [checkpoint]

epochs = 10

history = net.fit(train_generator, epochs=epochs, steps_per_epoch=150/batch_size, callbacks=callback_list)

testGene = testGenerator("./data/intel/test")
model = keras.models.load_model("./models/u2net_keras")
results = model.predict_generator(testGene,24,verbose=1)
saveResult("./data/intel/test/result",results)

history_dict = history.history
# acc = history_dict['accuracy']
loss = history_dict['loss']
# val_loss = history_dict['val_loss']
# val_accuracy = history_dict['val_accuracy']

# 借助 Matplotlib 绘制图像
# plt.plot(range(1, len(acc)+1), acc, 'b--')
plt.plot(range(1, len(loss)+1), loss, 'r-')
# plt.plot(range(1, len(val_loss)+1), val_loss, '-')
# plt.plot(range(1, len(val_accuracy)+1), val_accuracy, '--')

# 显示图例
plt.legend(['loss'])
plt.show()

