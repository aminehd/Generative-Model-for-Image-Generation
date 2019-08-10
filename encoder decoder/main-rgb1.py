import PIL.Image
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential, Model
from keras import regularizers
from scipy.misc import toimage
import numpy
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from Tkinter import *
import Tkinter
number_of_images = 100
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

        n = 7  # how many digits we will display
	plt.figure(figsize=(image_width, 7))
	print str(decoded_imgs[0])
	for i in range(n):
	    # display original
	    ax = plt.subplot(2, n, i + 1)
	    x_test[i] = x_test[i]
	    plt.imshow(x_test[i].reshape(image_width, image_width, 3))
	    #plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	
	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    decoded_imgs[i] = decoded_imgs[i] 
	    plt.imshow(decoded_imgs[i].reshape(image_width, image_width, 3))
	    #plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
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

# this is the size of our encoded representations
encoding_dim = 32  # 100 floats -> compression of factor 24.5, assuming the input is 1024 floats
image_width = 64
image_size = (image_width ** 2) * 3
# this is our input placeholder
input_img = Input(shape=(image_size,))
# "encoded" is the encoded representation of the input
encoded = Dense(600, activation='relu')(input_img)
encoded = Dense(400, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(400, activation='relu')(encoded)
decoded = Dense(600, activation='relu')(decoded)
decoded = Dense(image_size, activation='sigmoid')(decoded)
#decoded = Dense(3072, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)


# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = (autoencoder.layers[-3])(encoded_input)
decoder_layer = (autoencoder.layers[-2])(decoder_layer)
decoder_layer = (autoencoder.layers[-1])(decoder_layer)
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#from keras.datasets import cifar10
#import numpy as np
#(x_train, _), (x_test, _) = cifar10.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
####reading from file
path="data/celebB/"

x_train=numpy.empty([number_of_images,image_size])
x_test=numpy.empty([number_of_images,image_size])
i = 0
j = 0
for directory in os.listdir(path):
    x_train=numpy.empty([number_of_images,image_size])
    x_test=numpy.empty([number_of_images,image_size])
    image_list = os.listdir(path+directory)
    image_list = os.listdir(path+directory)[:number_of_images]
    for file in image_list:
        print(path+directory+"/"+file)
        img=PIL.Image.open(path+directory+"/"+file)
        print "our image format" + str(numpy.array(img))
        featurevector=numpy.array(img).flatten() #in my case the images dont have the same dimensions, so [:50] only takes the first 50 values
        if directory == 'train':
            print str(featurevector.shape)
            x_train[i] = featurevector
            i=i+1

        else:
            x_test[j] = featurevector
            j = j+1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print "sdvgSRGd"
print str(x_train[0])
print "sfds"
print str(x_train.shape)
print numpy.max(x_train)
#print "Y list"
print str(x_test.shape)
print numpy.max(x_test)
print x_train.shape
print x_test.shape

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#'''
autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
#'''
autoencoder.save_weights("model.h5")

#autoencoder.load_weights("model.h5")
# encode and decode some digits
# note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#encoded_imgs[0] = encoded_imgs[0] + 10

#encoded_imgs[1] = (encoded_imgs[0] * 1.0 + (encoded_imgs[3] * 2.0)) / 3.0

#decoded_imgs = decoder.predict(encoded_imgs)
'''
n = 7  # how many digits we will display
plt.figure(figsize=(20, 7))
print str(decoded_imgs[0])
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    x_test[i] = x_test[i]
    plt.imshow(x_test[i].reshape(image_width, image_width, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgs[i] = decoded_imgs[i] 
    plt.imshow(decoded_imgs[i].reshape(image_width, image_width, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
'''


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

#making the window
app = simpleapp_tk(None)
app.title('app')
app.mainloop()
####

