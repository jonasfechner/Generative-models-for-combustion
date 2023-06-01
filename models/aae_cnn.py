from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Lambda, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import tensorflow as tf


class AAE():
    def __init__(self, folder_name):
        self.folder_name = folder_name 

    #adversarial_autoencoder, encoder, decoder, discriminator, latent_model
    def labels_to_predict(self, labels):

        l = []
        min_label = np.genfromtxt('data/IR/normalised/labels_min.csv', delimiter=';')
        max_label = np.genfromtxt('data/IR/normalised/labels_max.csv', delimiter=';')
        
        for i, label in enumerate(labels):
            label_norm = (label - min_label[i]) / (max_label[i] - min_label[i]) 
            l.append(label_norm)
        return l



    def predict_from_labels(self, labels, channel_variable):

        autoencoder, _, decoder_model, _, latent_model = self.build_model()
        autoencoder.load_weights('weights/AAE_CNN/{}/autoencoder/{}.hdf5'.format(self.folder_name, channel_variable))
        latent_model.load_weights('weights/AAE_CNN/{}/latent/{}.hdf5'.format(self.folder_name, channel_variable))
        return decoder_model.predict(latent_model.predict(labels))

    def build_encoder(self):

        latent_shape = 300
        input_data = tf.keras.layers.Input(shape=(81, 241, 1))
        encoder = tf.keras.layers.Conv2D(128, kernel_size=(2,6), strides=(2, 2), activation='relu')(input_data)
        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(latent_shape + latent_shape)(encoder)
        latent_repr = tf.keras.layers.Dense(latent_shape)(encoder)
        return tf.keras.Model(input_data, latent_repr, name = 'Encoder')

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=300))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="tanh"))
        encoded_repr = Input(shape=(300, ))
        validity = model(encoded_repr)
        return Model(encoded_repr, validity, name = 'Discriminator')


    def build_decoder(self):

        decoder_input = tf.keras.layers.Input(shape=300)
        decoder = tf.keras.layers.Dense(64*11*31)(decoder_input)
        decoder = tf.keras.layers.Reshape((11, 31, 64))(decoder)
        decoder = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',activation='relu')(decoder)
        decoder = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',activation='relu')(decoder)
        decoder = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides = 2, padding='same')(decoder)
        decoder_output = tf.keras.layers.Cropping2D(cropping=((4, 3), (3, 4)))(decoder)
        return tf.keras.Model(decoder_input, decoder_output)


    def build_model(self):
        discriminator = self.build_discriminator()
        discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer = 'adam')  
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        img = Input(shape=(81, 241, 1) )
        encoded_repr = encoder(img)
        reconstructed_img = decoder(encoded_repr)
        discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer = 'adam')
        
        model_input = tf.keras.layers.Input(shape=3)
        model = tf.keras.layers.Dense(25, activation='relu')(model_input)
        model = tf.keras.layers.Dense(125, activation='relu')(model)
        model = tf.keras.layers.Dense(625, activation='relu')(model)
        model_output = tf.keras.layers.Dense(300)(model)
        latent_model = tf.keras.Model(model_input, model_output)
        latent_model.compile(loss='mean_absolute_error', optimizer='adam')
        
        return adversarial_autoencoder, encoder, decoder, discriminator, latent_model


    def prediction(self, channel_variable, X_train, X_test, verbose = 0):
        adversarial_autoencoder, encoder, _, discriminator, _ = self.build_model()
        # Adversarial ground truths
        valid = np.ones((X_train.shape[0], 1))
        fake = np.zeros((X_train.shape[0], 1))
        valid_test = np.ones((X_test.shape[0], 1))

        vall_loss = 10000000000

        def sample_prior(latent_dim, batch_size):
            return np.random.normal(size=(batch_size, latent_dim))

        for epoch in range(1000):
            
            latent_fake = encoder.predict(X_train, verbose = 0)
            latent_real = sample_prior(300, X_train.shape[0])
                            
            # Train the discriminator
            discriminator.fit(latent_real, valid, verbose = 0)
            discriminator.fit(latent_fake, fake, verbose = 0)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = adversarial_autoencoder.fit(
                X_train, 
                [X_train, valid], 
                verbose=verbose,
                validation_data = (X_test, [X_test, valid_test])
                )
            val_loss_new = g_loss.history['val_loss'][0]
            if val_loss_new < vall_loss: 
                #print(epoch, 'Weights updated')
                adversarial_autoencoder.save_weights('weights/AAE_CNN/{}/autoencoder/{}.hdf5'.format(self.folder_name, channel_variable))
                vall_loss = val_loss_new
                counter = 0
            if val_loss_new >= vall_loss:
                counter += 1
            if counter == 30: break

        return adversarial_autoencoder

    def latent_prediction(self, channel_variable, X_train, X_test,label_train, label_test):
        adversarial_autoencoder, encoder, _, _, latent_model = self.build_model()

        adversarial_autoencoder.load_weights('weights/AAE_CNN/{}/autoencoder/{}.hdf5'.format(self.folder_name, channel_variable))
        
        latent_train = encoder.predict(X_train, verbose = 0)
        latent_test = encoder.predict(X_test, verbose=0) 

        checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/AAE_CNN/{}/latent/{}.hdf5'.format(self.folder_name, channel_variable),
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    mode='min')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=50)

        latent_model.fit(label_train, latent_train, 
                            epochs=5000,
                            batch_size=16,
                            verbose = 0,
                            validation_data=(label_test, latent_test),
                            callbacks=[checkpoint, earlystopping])

        return latent_model

    def fit(self, channel_variable, train_data, test_data, label_train, label_test, verbose = 0, return_model = False):

        adversarial_autoencoder = self.prediction(channel_variable, train_data, test_data, verbose)
        latent_model = self.latent_prediction(channel_variable, train_data, test_data, label_train, label_test)

        if return_model:
            return adversarial_autoencoder, latent_model
        