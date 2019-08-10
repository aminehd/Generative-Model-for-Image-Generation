from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
from Tkinter import *
import Tkinter

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


class simpleapp_tk(Tkinter.Tk):
    global s1, s2, percentage
    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):

        button = Button(self, text="send", fg="red", command=self.callback)
        button.pack()
        # for i in range(encoding_dim):
        #    w = Scale(master, from_ = 0, to = 500, length = 600, orient = HORIZONTAL)
        #    w.pack()
        #    sliders.append(w)
        self.slider = Scale(self, from_=0, to=101, length=600, orient=HORIZONTAL)
        self.slider.pack()


        self.first_image = Entry(self)
        self.first_image.pack()
        self.second_image = Entry(self)
        self.second_image.pack()
    def callback(self):
        s1 = (int)(self.first_image.get())
        s2 = (int)(self.second_image.get())
        percentage = self.slider.get()
        print "reduced features are" + str(reduced_features)
        decoded_imgs = decoder.predict(encoded_imgs)
        main_output = decoded_imgs[selected]

        # Feeding user features into the network
        # encoded_imgs[selected] = reduced_features
        # decoded_imgs = decoder.predict(encoded_imgs)
        f1 = encoded_imgs[s1]
        f2 = encoded_imgs[s2]
        ## reading the percentage
        print "percentage " + str(percentage)

        encoded_imgs[0] = f1.astype('float32') * ((float)(percentage - 1) / 100) + f2.astype('float32') * (
        (float)(101 - percentage) / 100)
        print "encoded image" + str()
        print str(((float)(percentage - 1) / 100)) + "first percentage"
        print str(((float)(101 - percentage) / 100)) + "second percentage"

        decoded_imgs = decoder.predict(encoded_imgs)

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # print str(x_test[i]) + " x_test[i]"

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # print str(decoded_imgs[i]) + " decoded_imgs[i]"
            # desplay the main image
            # display original

            #
        plt.show()

    #master.destroy()

# list of all sliders
sliders = []
s1 = 1
s2 = 2
percentage = 1
reduced_features = []


# Button handler for B



# Making a slider
master = Tk()

# A button for Tkinter

# this is our input placeholder
input_img = Input(shape=(784,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train[:20]
# x_test = x_test[:20]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)  ###print it
print "shape" + str(encoded_imgs.shape)
# normalizing list variables

# selected image is the 0
selected = 0
selected_img = encoded_imgs[selected]
minimum = min(selected_img)
maximum = max(selected_img)
normalizer = sum(selected_img) / len(selected_img)


# The normalizer funcgtion
def normalize(a):
    return a * 100


def de_normalize(a):
    return a / 100


normalized_vector = map(normalize, selected_img)
print selected_img
print normalized_vector
# Set the values on the slider
# for i in range(encoding_dim):
#    sliders[i].set(normalized_vector[i])
#    sliders[i].pack()


#making the window
app = simpleapp_tk(None)
app.title('app')
app.mainloop()
####

