from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Conv2D,PReLU,MaxPooling2D,BatchNormalization,Reshape,Conv2DTranspose

class test(Model):
    
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(10, activation="softmax"))
        

        
    def call(self, input):
        
        x = Flatten()(input)
        output = self.model(x)
        
        return output

class Net_v1(Model):
    
    def __init__(self):
        super().__init__()
        self.dense512 = Dense(512, activation="relu")
        self.dense256 = Dense(256, activation="relu")
        self.dense128 = Dense(128, activation="relu")
        self.dense64 = Dense(64, activation="relu")
        self.dense10 = Dense(10, activation="softmax")
        
    def call(self, input):
        
        x = Flatten()(input)
        x = self.dense512(x)
        x = self.dense256(x)
        x = self.dense128(x)
        x = self.dense64(x)
        output = self.dense10(x)
        
        return output
    
class ConvTranspose(Model):
    
    def __init__(self, out_channel, kernel_size=3, strides=1, padding=True, output_padding=None):
        super().__init__()
        if padding:
            self.conv = Conv2DTranspose(out_channel, kernel_size=kernel_size, strides=strides, padding='same', output_padding=output_padding)
        else:
            self.conv = Conv2DTranspose(out_channel, kernel_size=kernel_size, strides=strides, output_padding=output_padding)
        self.prelu = PReLU('ones')
        self.batch = BatchNormalization()
        
    def call(self, input):
        
        x = self.conv(input)
        x = self.prelu(x)
        output = self.batch(x)
        return output
    
class Conv(Model):
    
    def __init__(self, out_channel, kernel_size=3, strides=1, padding=True):
        super().__init__()
        if padding:
            self.conv = Conv2D(out_channel, kernel_size=kernel_size, strides=strides, padding='same')
        else:
            self.conv = Conv2D(out_channel, kernel_size=kernel_size, strides=strides)
        self.prelu = PReLU('ones')
        self.batch = BatchNormalization()
        
    def call(self, input):
        
        x = self.conv(input)
        x = self.prelu(x)
        output = self.batch(x)
        return output
    
class Net_v2(Model):
    
    def __init__(self):
        super().__init__()
        self.conv12 = Conv(12)
        self.conv24 = Conv(24)
        self.conv56 = Conv(56)
        self.conv128 = Conv(128)
        self.conv256 = Conv(256)
        self.conv512 = Conv(512)
        self.conv786 = Conv(786,padding=False)
        self.maxpool = MaxPooling2D()
        self.flatten = Flatten()
        self.dense10 = Dense(10, activation="softmax")
        
    def call(self, input):
        
        x = self.conv12(input)
        x = self.maxpool(x)
        x = self.conv24(x)
        x = self.maxpool(x)
        x = self.conv56(x)
        x = self.maxpool(x)
        x = self.conv128(x)
        x = self.conv256(x)
        x = self.conv512(x)
        x = self.conv786(x)
        x = self.flatten(x)
        output = self.dense10(x)
        
        return output
    
class Net_v3(Model):
    
    def __init__(self):
        super().__init__()
        self.conv32 = Conv(32)
        self.maxpool = MaxPooling2D()
        self.conv64 = Conv(64)
        self.conv128 = Conv(128, padding=False)
        self.conv256 = Conv(256, padding=False)
        self.conv512 = Conv(512, padding=False)
        self.flatten = Flatten()
        self.dense10 = Dense(10, activation="softmax")
        
    def call(self, input):
        
        x = self.conv32(input)
        x = self.maxpool(x)
        x = self.conv64(x)
        x = self.maxpool(x)
        x = self.conv128(x)
        x = self.conv256(x)
        x = self.conv512(x)
        x = self.flatten(x)
        output = self.dense10(x)
        
        return output
    
class Encoder(Model):
    
    # def __init__(self):
    #     super().__init__()
    #     self.flatten = Flatten()
    #     self.dense512 = Dense(512, activation="relu")
    #     self.dense256 = Dense(256, activation="relu")
    #     self.dense128 = Dense(128, activation="relu")
    #     self.dense64 = Dense(64, activation="relu")
    #     self.dense10 = Dense(10, activation="softmax")
        
        
    # def call(self, input):
    #     x = self.flatten(input)
    #     x = self.dense512(x)
    #     x = self.dense256(x)
    #     x = self.dense128(x)
    #     x = self.dense64(x)
    #     output = self.dense10(x)
        
    #     return output
    
    def __init__(self):
        super().__init__()
        self.conv16 = Conv(16)
        self.conv32 = Conv(32)
        self.conv64 = Conv(64)
        self.conv128 = Conv(128)
        
    def call(self, input):
        x = self.conv16(input)
        x = self.conv32(x)
        x = self.conv64(x)
        output = self.conv128(x)
        
        return output
    
class Decoder(Model):
    
    # def __init__(self):
    #     super().__init__()
    #     self.dense64 = Dense(64, activation="relu")
    #     self.dense128 = Dense(128, activation="relu")
    #     self.dense256 = Dense(256, activation="relu")
    #     self.dense512 = Dense(512, activation="relu")
    #     self.dense = Dense(28*28)
    #     self.reshape = Reshape((28,28,1))
        
    # def call(self, input):
    #     x = self.dense64(input)
    #     x = self.dense128(x)
    #     x = self.dense256(x)
    #     x = self.dense512(x)
    #     x = self.dense(x)
    #     output = self.reshape(x)
        
    #     return output
    
    def __init__(self, channels=3):
        super().__init__()
        self.convtranspose64 = ConvTranspose(64)
        self.convtranspose32 = ConvTranspose(32)
        self.convtranspose16 = ConvTranspose(16)
        self.convtranspose= ConvTranspose(channels)
        
    def call(self, input):
        x = self.convtranspose64(input)
        x = self.convtranspose32(x)
        x = self.convtranspose16(x)
        output = self.convtranspose(x)
        
        return output
        
            
class Main_Net(Model):
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def call(self, input):
        encoder_out = self.encoder(input)
        output = self.decoder(encoder_out)
        
        return output

if __name__ == '__main__':
    
    import numpy as np
    
    net = test()
    net.build(input_shape=(None,32,32,3))
    net.summary()
    test = np.random.rand(1,32,32,3)
    test_y = net.call(test)
    print(test_y.shape)