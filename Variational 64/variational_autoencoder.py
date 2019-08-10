'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import matplotlib.image as matimg

import glob
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from Tkinter import *
import Tkinter
class simpleapp_tk(Tkinter.Tk):
    global s1, s2, percentage, x_test
    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.fig = []
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
        self.cut = Entry(self)
        self.cut.pack()
    def callback(self):
        global faces, encoded_faces, x_test_encoded, x_test
        if self.fig != []:
            plt.close(self.fig)
        s1 = (int)(self.first_image.get())
        s2 = (int)(self.second_image.get())
        start_cut = (int)(self.cut.get())
        percentage = self.slider.get()
        grid_x = np.linspace(-1.6, 1.6, n)
        grid_y = np.linspace(-5.5, 5.5, n)

        face_size = 64
        selected = [7, 19, 26, 96, 9, 2, 3, 5, 87, 65]
        cut_encoded = x_test_encoded[start_cut:start_cut + 10]
        cut_test = x_test[start_cut: start_cut + 10]
        print str(cut_encoded.shape) + "cutencoded shape and x_test_encoded is" + str(x_test_encoded.shape)
        # code to create a result.png with generated faces
        faces = []
        encoded_faces = []
        for i, yi in enumerate(cut_encoded):
            z_sample = np.array([[yi[0], yi[1]]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

            print "z_sample shape" + str(z_sample.shape)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            x_decoded = x_decoded
            #print x_decoded.shape
            encoded_faces.append(z_sample)
            face = x_decoded[0].reshape(face_size, face_size, 3)
            faces.append(face)

        print "hi"
        self.fig = plt.figure(figsize=(10, 3))

        for i, yi in enumerate(cut_encoded):
            print "hi"
            face = faces[i]

            ax = plt.subplot(3, 10, i+1)
            plt.imshow(face)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax1 = plt.subplot(3, 10, i + 10 + 1)
            plt.imshow(cut_test[i].reshape(face_size, face_size, 3))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
        f1 = encoded_faces[s1]
        f2 = encoded_faces[s2]
        a = f1.astype('float32') * ((float)(percentage - 1) / 100) + f2.astype('float32') * ((float)(101 - percentage) / 100)
        x_decoded = generator.predict(a, batch_size=batch_size)
        x_decoded = x_decoded
        face = x_decoded[0].reshape(face_size, face_size, 3)
        ax = plt.subplot(3, 10, 24)
        plt.imshow(face)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
image_width = 64
number_of_pictures = 9000
num_testing_examples = 1000
image_vector_size = image_width * image_width * 3 #this will be img_width X img_height X 3 (for rgb)

batch_size = 100
original_dim = image_vector_size #this is same as image_vector_size
latent_dim = 2
intermediate_dim = 1024
nb_epoch = 300
epsilon_std = 1.0

#following code loads data from local repo
#directory should have two subfolders, with name train and test
def load_image(path):
    img = matimg.imread(path)
    return img

def load_local_data(path):
    # Get image paths
    paths = glob.glob(os.path.join(path + "/train", "*.jpg"))
    # Load images
    X_train = np.array( [ load_image(p) for p in paths ] )[:number_of_pictures]

    # Get image paths
    paths = glob.glob(os.path.join(path + "/test", "*.jpg"))
    # Load images
    X_test = np.array( [ load_image(p) for p in paths ] )[:number_of_pictures]

    return X_train, X_test

def load_test(path):
    paths = glob.glob(os.path.join(path + "/test", "*.jpg"))
    X_test = np.array( [ load_image(p) for p in paths ] )[:num_testing_examples]
    return X_test
#building model
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

#sample z based on z = mu + sdev * epsilon
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# modified calculation of loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss +  kl_loss

# vae model
vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# if want to train with data from local repo
path="data/celebB/"
(x_train, x_test) = load_local_data(path)
#x_test = load_test(path)
#x_train = np.ones(number_of_pictures)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#to train the model

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

vae.save_weights("model.h5")

#vae.load_weights("model.h5")


#to load the already trained model weights
#vae.load_weights('vae_face_64_300.h5')

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(10, 10))
dummy_c = np.zeros((x_test_encoded.shape[0]))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=dummy_c)
plt.colorbar()
plt.show()

# build a face generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the faces
n = 10 # figure with 10x10 faces
face_size = 64
#figure = np.zeros((face_size * n, face_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian

# change values below to generate new images

#making the window
app = simpleapp_tk(None)
app.title('app')
app.mainloop()
####

