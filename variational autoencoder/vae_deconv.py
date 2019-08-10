'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.image as matimg

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import os
import glob
from Tkinter import *
import Tkinter


class simpleapp_tk(Tkinter.Tk):
    global s1, s2, percentage, x_test_encoded, batch_size, faces, x_test
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

        # code to create a result.png with generated faces
        print "hi"
        plt.figure(figsize=(10, 2))
        for i, yi in enumerate(x_test_encoded):
            print "hi"

            face = faces[i]

            ax = plt.subplot(2, 10, i+1)
            plt.imshow(face)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax1 = plt.subplot(2, 10, i + 10 + 1)
            plt.imshow(x_test[i].reshape(face_size, face_size, 3))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
        plt.show()
        #figure[i * face_size: (i + 1) * face_size,
         #      j * face_size: (j + 1) * face_size] = face
        '''
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
        '''

batch_size = 25
img_rows, img_cols, img_chns = 32, 32, 3
number_of_pictures = 250
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
# input image dimensions

# number of convolutional filters to use
nb_filters = 32
# convolution kernel size
nb_conv = 3


if K.image_dim_ordering() == 'th':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
nb_epoch = 400

x = Input(batch_shape=(batch_size,) + original_img_size)
conv_1 = Convolution2D(4, 2, 2, border_mode='same', activation='relu')(x)
conv_2 = Convolution2D(nb_filters, 2, 2,
                       border_mode='same', activation='relu',
                       subsample=(2, 2))(conv_1)
#output shape == (batch_size,
conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                       border_mode='same', activation='relu',
                       subsample=(1, 1))(conv_2)
conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                       border_mode='same', activation='relu',
                       subsample=(1, 1))(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(nb_filters * 16 * 16, activation='relu')

if K.image_dim_ordering() == 'th':
    output_shape = (batch_size, nb_filters, 16, 16)
else:
    output_shape = (batch_size, 16, 16, nb_filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                   output_shape,
                                   border_mode='same',
                                   subsample=(1, 1),
                                   activation='relu')
decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                   output_shape,
                                   border_mode='same',
                                   subsample=(1, 1),
                                   activation='relu')
if K.image_dim_ordering() == 'th':
    output_shape = (batch_size, nb_filters, 33, 33)
else:
    output_shape = (batch_size, 33, 33, nb_filters)
decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2,
                                          output_shape,
                                          border_mode='valid',
                                          subsample=(2, 2),
                                          activation='relu')
decoder_mean_squash = Convolution2D(img_chns, 2, 2,
                                    border_mode='valid',
                                    activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean_squash)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# train the VAE on MNIST digits
#(x_train, _), (x_test, y_test) = mnist.load_data()

# if want to train with data from local repo
path="data/celebC/"
(x_train, x_test) = load_local_data(path)
print "x_train shape = " + str(x_train.shape)
print "original" + str((x_train.shape[0],) + original_img_size)

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

vae.save_weights("model.h5")

#vae.load_weights("model.h5")
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# display a 2D manifold of the digits
n = 32  # figure with 15x15 digits
digit_size = 32
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

#for i, yi in enumerate(grid_x):
#    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
#        x_decoded = generator.predict(z_sample, batch_size=batch_size)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#        figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit
#
#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')
#plt.show()
face_size = 32
x_test_encoded = x_test_encoded[:10]
x_test = x_test[:10]
# code to create a result.png with generated faces
faces = []
for i, yi in enumerate(x_test_encoded):
    z_sample = np.array([[yi[0], yi[1]]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

    print "z_sample shape" + str(z_sample.shape)
    x_decoded = generator.predict(z_sample, batch_size=batch_size)
    x_decoded = x_decoded
    #print x_decoded.shape
    face = x_decoded[0].reshape(face_size, face_size, 3)
    faces.append(face)
'''
plt.figure(figsize=(10, 2))
for i, yi in enumerate(x_test_encoded):
    z_sample = np.array([[yi[0], yi[1]]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

    print "z_sample shape" + str(z_sample.shape)
    x_decoded = generator.predict(z_sample, batch_size=batch_size)
    x_decoded = x_decoded
    #print x_decoded.shape
    face = x_decoded[0].reshape(face_size, face_size, 3)

    ax = plt.subplot(2, 10, i+1)
    plt.imshow(face)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax1 = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_test[i].reshape(face_size, face_size, 3))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
        #figure[i * face_size: (i + 1) * face_size,
         #      j * face_size: (j + 1) * face_size] = face
'''
#plt.figure(figsize=(10, 10))
#plt.savefig('result.png')
#making the window
app = simpleapp_tk(None)
app.title('app')
app.mainloop()
####

