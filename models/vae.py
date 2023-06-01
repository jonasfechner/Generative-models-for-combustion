import tensorflow as tf
import numpy as np
import tensorflow
from tensorflow import keras
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import os

class VAE():
    def __init__(self, folder_name):
        self.folder_name = folder_name
        

    def labels_to_predict(self, labels):

        l = []
        min_label = np.genfromtxt('data/IR/normalised/labels_min.csv', delimiter=';')
        max_label = np.genfromtxt('data/IR/normalised/labels_max.csv', delimiter=';')
        
        for i, label in enumerate(labels):
            label_norm = (label - min_label[i]) / (max_label[i] - min_label[i]) 
            l.append(label_norm)
        return l

    def predict_from_labels(self, labels, channel_variable):

        autoencoder, _, decoder_model,  latent_model = self.build_model()
        autoencoder.load_weights('weights/VAE/{}/autoencoder/{}.hdf5'.format(self.folder_name, channel_variable))
        latent_model.load_weights('weights/VAE/{}/latent/{}.hdf5'.format(self.folder_name, channel_variable))
        return decoder_model.predict(latent_model.predict(labels))
       

    def build_model(self):
        latent_shape = 300
        input_data = tensorflow.keras.layers.Input(shape=(81, 241, 1))

        encoder = tensorflow.keras.layers.Conv2D(128, kernel_size=(2,6), strides=(2, 2), activation='relu')(input_data)
        encoder = tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(2,6), strides=(2, 2), activation='relu')(encoder)
        encoder = tensorflow.keras.layers.Flatten()(encoder)
        encoder = tensorflow.keras.layers.Dense(latent_shape + latent_shape)(encoder)

        def sample_latent_features(distribution):
            distribution_mean, distribution_variance = distribution
            batch_size = tensorflow.shape(distribution_variance)[0]
            random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
            return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

        distribution_mean = tensorflow.keras.layers.Dense(latent_shape, name='mean')(encoder)
        distribution_variance = tensorflow.keras.layers.Dense(latent_shape, name='log_variance')(encoder)
        latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])
        encoder_model = tensorflow.keras.Model(input_data, latent_encoding)

        decoder_input = tensorflow.keras.layers.Input(shape=latent_shape)
        decoder = tensorflow.keras.layers.Dense(64*11*31)(decoder_input)
        decoder = tensorflow.keras.layers.Reshape((11, 31, 64))(decoder)
        decoder = tensorflow.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',activation='relu')(decoder)
        decoder = tensorflow.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',activation='relu')(decoder)
        decoder = tensorflow.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides = 2, padding='same')(decoder)
        decoder_output = tensorflow.keras.layers.Cropping2D(cropping=((4, 3), (3, 4)))(decoder)
        decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)

        encoded = encoder_model(input_data)
        decoded = decoder_model(encoded)
        autoencoder = tensorflow.keras.models.Model(input_data, decoded)

        def get_loss(distribution_mean, distribution_variance):
            
            def get_reconstruction_loss(y_true, y_pred):
                reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
                reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
                return reconstruction_loss_batch*241*81
            
            def get_kl_loss(distribution_mean, distribution_variance):
                kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(distribution_variance)
                kl_loss_batch = tensorflow.reduce_mean(kl_loss)
                return kl_loss_batch*(-0.5)
            
            def total_loss(y_true, y_pred):
                reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
                kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
                return reconstruction_loss_batch + kl_loss_batch
            
            return total_loss

        autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')

        model_input = tensorflow.keras.layers.Input(shape=3)
        model = tensorflow.keras.layers.Dense(25, activation='relu')(model_input)
        model = tensorflow.keras.layers.Dense(125, activation='relu')(model)
        model = tensorflow.keras.layers.Dense(625, activation='relu')(model)
        model_output = tensorflow.keras.layers.Dense(300)(model)

        latent_model = tensorflow.keras.Model(model_input, model_output)
        latent_model.compile(loss='mean_absolute_error', optimizer='adam')
        
        return autoencoder, encoder_model, decoder_model, latent_model


    def prediction(self, channel_variable, train_data, test_data):
        autoencoder, _, _, _ = self.build_model()
        
        checkpoint = keras.callbacks.ModelCheckpoint("weights/VAE/{}/autoencoder/{}.hdf5".format(self.folder_name, channel_variable),
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=True,
                            mode='min')
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        autoencoder.fit(train_data, train_data, 
                    epochs=500,
                    batch_size=16,
                    verbose = 0,
                    validation_data=(test_data, test_data),
                    callbacks=[checkpoint, earlystopping])


    def latent_prediction(self, channel_variable, train_data, test_data, label_train, label_test):

        autoencoder, encoder_model, _, latent_model = self.build_model()
        
        autoencoder.load_weights('weights/VAE/{}/autoencoder/{}.hdf5'.format(self.folder_name, channel_variable))
        
        latent_train = encoder_model.predict(train_data, verbose = 0)
        latent_test = encoder_model.predict(test_data, verbose=0)

        checkpoint = keras.callbacks.ModelCheckpoint("weights/VAE/{}/latent/{}.hdf5".format(self.folder_name, channel_variable),
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    mode='min')
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=50)

        latent_model.fit(label_train, latent_train, 
                        epochs=5000,
                        batch_size=16,
                        verbose = 0,
                        validation_data=(label_test, latent_test),
                        callbacks=[checkpoint, earlystopping])

    def fit(self, train_data, test_data, label_train, label_test):

        self.prediction(train_data, test_data)
        self.latent_prediction(train_data, test_data, label_train, label_test)



