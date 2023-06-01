import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from matplotlib import pyplot as plt

max_Q = 8106652700.0

import numpy
from numpy import genfromtxt
train_images = genfromtxt('set2.csv', delimiter=';')
train_images = numpy.reshape(train_images, (-1, 75, 300,1))
train_images = train_images * max_Q

train_images = (train_images - max_Q/2) / (max_Q/2)

# latent dimension of the random noise
LATENT_DIM = 512 
# weight initializer for G per DCGAN paper 
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) 
# number of channels, 1 for gray scale and 3 for color images
CHANNELS = 1  

noise_dim = 512
# dense 5*5*512, relu, reshape (5,5,512), Conv2DTranspose, batchnormalisation, relu, Conv2DTranspose, batch, relu, Conv2DTranspose
def build_generator():

    model = Sequential(name="generator")
    model.add(layers.Dense(5 * 5 * 512, input_dim=LATENT_DIM))
    model.add(layers.ReLU())
    model.add(layers.Reshape((5, 5, 512)))
    model.add(layers.Conv2DTranspose(256, (4, 4), 
                                     strides=(1, 4), 
                                     padding="same", 
                                     use_bias=False, 
                                     kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization()) # Margaret exp: add back bach norm
    model.add((layers.ReLU()))
    model.add(layers.Conv2DTranspose(128, (4, 4), 
                                     strides=(5, 5), 
                                     padding="same", 
                                     use_bias=False, 
                                     kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization()) # Margaret exp: add back bach norm
    model.add((layers.ReLU()))
    model.add(layers.Conv2DTranspose(64, (4, 4), 
                                     strides=(3, 3), 
                                     padding="same", 
                                     use_bias=False, 
                                     kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization()) # Margaret exp: add back bach norm
    model.add((layers.ReLU()))
    model.add(layers.Conv2D(CHANNELS, (4, 4), padding="same", activation="tanh"))
    
    return model


def build_critic(height, width, depth, alpha=0.2):

    model = Sequential(name="critic")
    input_shape = (height, width, depth)
    model.add(layers.Conv2D(64, (4, 4), padding="same", 
                            strides=(2, 2),
                            input_shape=input_shape
                            ))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Conv2D(128, (4, 4), padding="same", 
                            strides=(2, 2), 
                            )) # UPDATE for WGAN
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Conv2D(128, (4, 4), padding="same", 
                            strides=(2, 2),
                            )) # UPDATE for WGAN
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3)) 
    model.add(layers.Dense(1, activation="linear"))

    return model         



class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                
                # Calculate the discriminator loss using the fake and real image logits
                d_fake = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)[1]
                d_real = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)[2]
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)[0]
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        return {"d_loss": d_loss, "g_loss": g_loss, "d real": d_real, "d_fake": d_fake}




class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 100 == 0:
            self.model.generator.save('../weights/WGAN/generator_{:03d}.h5'.format(epoch)) 

            random_latent_vectors = numpy.random.normal(0,1, (70,512))

            prediction = numpy.reshape(self.model.generator.predict(random_latent_vectors), (-1, 81*241))
            prediction_image = numpy.reshape(prediction, (-1, 75, 300))
            def plotting(plots, pred):
                fig, axs = plt.subplots(plots)
                for i in range(plots):
                    axs[i].imshow(pred[i])
                plt.show()
            
            
            #plotting(5, prediction_image)
        #current_loss = logs.get("g_loss_fn")
        #with open('log.csv','a') as fd:
        #    fd.write(str(current_loss))
        
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return [fake_loss - real_loss, fake_loss, real_loss]


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitor(num_img=3, latent_dim=noise_dim)


d_model = build_critic(81, 241, 1) 
g_model = build_generator() 


# Get the wgan model
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=noise_dim,
    discriminator_extra_steps=3,
)

# Compile the wgan model
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)
BATCH_SIZE = 64
epochs = 200000
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk], verbose = 1)



