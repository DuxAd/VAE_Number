import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Lambda, Concatenate, Dropout, BatchNormalization, Reshape, Conv2D, Input, LeakyReLU, Flatten, Dense, Activation, Conv2DTranspose
from keras.optimizers import Adam
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import keras.backend as K
import pickle
# %matplotlib qt #for better 3D Graph on Spyder

class Autoencoder:
    def __init__(self, x1,x2,x3,x4,x5,x6,x7,x8):
        self.input_dim = x1
        self.encoder_conv_filters = x2
        self.encoder_conv_kernel_size = x3
        self.encoder_conv_strides = x4
        self.decoder_conv_t_filters = x5
        self.decoder_conv_t_kernel_size = x6
        self.decoder_conv_t_strides = x7
        self.z_dim = x8
        
    ####### Encoder definition #######
    def f_encoder(self):
        self.encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = self.encoder_input
        for i in range(len(self.encoder_conv_filters)):
            conv_layer = Conv2D(filters = self.encoder_conv_filters[i],
                                 kernel_size = self.encoder_conv_kernel_size[i],
                                 strides = self.encoder_conv_strides[i],
                                 padding = 'same',
                                 name='Encoder_Conv2D_' + str(i)
                             )
            
            x = conv_layer(x)
            # x = BatchNormalization()(x)
            x = LeakyReLU(name='LeakyRelu_'+str(i))(x)
            # x = Dropout(rate=0.25)(x)
            
        self.Out_shape = x.shape[1:]
        x = Flatten()(x)
        x = Dense(64)(x)
        x = Activation("relu")(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        
        def sampling(args):
            mu, log_var = args
            
            epsilon = tf.random.normal(shape=tf.shape(mu)) 
            return mu + tf.exp(log_var/2)*epsilon
        
        self.mu_logvar = Concatenate(name="mu_logvar")([self.mu, self.log_var])
        
        self.encoder_output=Lambda(sampling, output_shape= (self.z_dim,), name='encoder_output')([self.mu, self.log_var])
        
        self.encoder = Model(self.encoder_input, [self.encoder_output, self.mu_logvar])
    
    ####### Decoder definition #######
    def f_decoder(self):
        self.decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(self.Out_shape))(self.decoder_input)
        x = Reshape(self.Out_shape)(x)

        for i in range(len(self.decoder_conv_t_filters)):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = 'same',
                name = 'Decoder_Conv2DTrans_' + str(i))
            x = conv_t_layer(x)
    
            if i < len(self.decoder_conv_t_filters) - 1:
                x = LeakyReLU()(x)
        x = Activation('sigmoid')(x)


        self.decoder_output = x
        self.decoder= Model(self.decoder_input, self.decoder_output)
    
    ####### Full Model definition #######
    def f_FullModel(self, learning_rate):
        ## Building the full model
        self.f_encoder()
        self.f_decoder()
        model_input = self.encoder_input
        z, mu_log_var = self.encoder(model_input)
        model_output = self.decoder(z)
        self.model = Model(model_input, [model_output, mu_log_var], name='vae') 
        
        ## Compilation
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss= [self.vae_r_loss, self.vae_kl_loss])
        
    ####### Loss functions #######
    def vae_r_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        r_loss_factor = 1
        r_loss = tf.keras.losses.binary_crossentropy(x, y) 
        return r_loss_factor*r_loss
        
    def vae_kl_loss(self, y_true, y_pred):
        mu = y_pred[:, :self.z_dim]
        log_var = y_pred[:, self.z_dim:]
        kl_loss = -0.5*tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        return kl_loss * .003
    
    ####### Training #######
    def f_train(self, npimages, npimages_test):
        from keras.callbacks import EarlyStopping
        self.history = self.model.fit(
            x=npimages, y=[npimages, np.zeros((len(npimages), self.z_dim*2))] , #[npimages,npimages,npimages], #npimages
            validation_data=(npimages_test, [npimages_test, np.zeros((len(npimages_test), self.z_dim*2))]),
            batch_size = 128,
            shuffle = True,
            epochs=15,
            callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
            )
        
## Data loading
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
npimages = np.array([image.numpy()/255 for image,_ in ds_train])
npimages_test = np.array([image.numpy()/255 for image,_ in ds_test])

## Model Creation
new = 0
load = 1

if new:
    AE = Autoencoder(
         (28,28,1),
         [32,64,64],
         [3,3,3],
         [1,2,2],
         [64,64,32,1],
         [3,3,3,3],
         [2,2,1,1],
         3)
    AE.f_FullModel(learning_rate = 0.0005)
    AE.f_train(npimages, npimages_test)

    ## Plotting the loss
    plt.plot(AE.history.history['loss'], label='Total Loss')
    plt.legend()
    plt.show()
elif load :
    with open('autoencoder_attributes.pkl', 'rb') as f:
        loaded_attributes = pickle.load(f)

    AE = Autoencoder(
        x1=loaded_attributes['input_dim'],
        x2=loaded_attributes['encoder_conv_filters'],
        x3=loaded_attributes['encoder_conv_kernel_size'],
        x4=loaded_attributes['encoder_conv_strides'],
        x5=loaded_attributes['decoder_conv_t_filters'],
        x6=loaded_attributes['decoder_conv_t_kernel_size'],
        x7=loaded_attributes['decoder_conv_t_strides'],
        x8=loaded_attributes['z_dim']
    )

    AE.Out_shape = loaded_attributes['Out_shape']
    AE.f_FullModel(learning_rate = 0.0005)
    AE.model.load_weights('MyModel.keras')

    AE.model.summary()

## Image reconstruction
fig_1, axs_1 = plt.subplots(3, 5)
fig_2, axs_2 = plt.subplots(3, 5)
prediction = AE.model.predict(npimages[:15])
for i in range(15):
    ima = npimages[i]

    axs_1[i//5,i%5].imshow(ima)
    axs_2[i//5,i%5].imshow(prediction[0][i])
    
## New Image generation
fig_3, axs_3 = plt.subplots(3, 5)
nb = 15
latent_vector = np.array([np.random.normal(loc=0, scale=1, size=[1,AE.z_dim]) for i in range(nb)]).reshape(nb, AE.z_dim)
decoded=AE.decoder.predict(latent_vector).reshape(nb,28,28,1)
for i in range(nb):
    axs_3[i//5,i%5].imshow(decoded[i])

### Latent space
(images_np, labels_np) = tfds.as_numpy(tfds.load(
    'mnist',
    split='train',
    batch_size=-1, # Load a single batch
    as_supervised=True,
))

fig_4 = plt.figure()
ima = images_np[:15000]
label = labels_np[:15000]
z, mu_log_var = AE.encoder.predict(ima/255)

if AE.z_dim == 2:
    ## For 2D latent space
    fig_4 = plt.figure()
    plt.scatter(z[:, 0], z[:, 1], c=label, cmap = 'tab10',s=1)
    plt.colorbar()
    plt.xlabel('Z_0')
    plt.ylabel('Z_1')
    plt.title('Z distribution in the latent space')
    plt.show()
elif AE.z_dim == 3:
    ## For 3D latent space
    ax = fig_4.add_subplot(projection='3d')
    for i in range(10):
        ax.scatter(z[label==i, 0], z[label==i, 1], z[label==i, 2], label=i,s=1)
    ax.set_xlabel('Z_0')
    ax.set_ylabel('Z_1')
    ax.set_zlabel('Z_2')
    ax.set_title('Z distribution in the latent space')
    ax.legend(title='Number', 
              loc='best', 
              markerscale=10) 
    
    plt.show()
    
## Plotting the numbers along a line in the latent space
fig_5, axs_5 = plt.subplots(5, 5)
start=-1
end=1
inter=25

latent_vector = np.array([np.full((1, AE.z_dim), start + i * (end - start) / (inter-1) ) for i in range(inter)]).reshape(inter, AE.z_dim)
decoded=AE.decoder.predict(latent_vector).reshape(inter,28,28,1)
for i in range(inter):
    axs_5[i//5,i%5].imshow(decoded[i])


## Saving
save=1
if save:
    AE.model.save("MyModel.keras")
    attributes = {
        'input_dim': AE.input_dim,
        'encoder_conv_filters': AE.encoder_conv_filters,
        'encoder_conv_kernel_size': AE.encoder_conv_kernel_size,
        'encoder_conv_strides': AE.encoder_conv_strides,
        'decoder_conv_t_filters': AE.decoder_conv_t_filters,
        'decoder_conv_t_kernel_size': AE.decoder_conv_t_kernel_size,
        'decoder_conv_t_strides': AE.decoder_conv_t_strides,
        'z_dim': AE.z_dim,
        'Out_shape': AE.Out_shape,
    }
    with open('autoencoder_attributes.pkl', 'wb') as f:
        pickle.dump(attributes, f)

