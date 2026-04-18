###########################
## Author: Ayse Dincer
## Date: Feb 18 2020
## Script for generating AD-AE embeddings
###########################

import pandas as pd
import numpy as np

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model

import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import seed
from tensorflow import set_random_seed

#Read input data
input_file = 'TCGA_GBM_and_LGG_PREPROCESSED_RNASEQ_EXPRESSION_500_kmeans.tsv'
input_df = pd.read_csv(input_file, sep = '\t', index_col = 0)
input_df = pd.DataFrame(input_df.values, index = input_df.index.astype(str))
print("Input df ", input_df.shape)
print("Input df ", input_df.head(5))
print(input_df)

#Read sex labels
filename = "TCGA_GBM_and_LGG_SEX_LABELS.tsv"
pheno_df = pd.read_csv(filename, sep = '\t', index_col = 0)
pheno_df = pheno_df.dropna()
new_labels = [np.where(['male', 'female'] == i)[0][0] for i in pheno_df.values]
pheno_df = pd.DataFrame(new_labels, index = pheno_df.index)

#Detct common samples
common_samples = np.intersect1d(input_df.index.values, pheno_df.index.values)
pheno_df = pheno_df.loc[common_samples]
input_df = input_df.loc[common_samples]

X = input_df
Z = pheno_df

#Train and test splits
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2, random_state=12345)

#Standardize the data
scaler = StandardScaler().fit(X_train)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X_train = X_train.pipe(scale_df, scaler) 
X_test = X_test.pipe(scale_df, scaler)

print("X train ", X_train.shape)
print("Z train ", Z_train.shape)
print("X test ", X_test.shape)
print("Z test ", Z_test.shape)

#Class for adversarial training
class AdversarialDeconfoundingAutoencoder(object):

    def __init__(self, n_features, latent_dim, lambda_val, random_seed):
        
        #Set random seeds
        seed(123456 * random_seed)
        set_random_seed(123456 * random_seed)

        #Set values
        self.lambda_val = lambda_val
        self.latent_dim = latent_dim
        
        #Define inputs
        ae_inputs = Input(shape=(n_features,)) 
        adv_inputs = Input(shape=(latent_dim,))
        
         #Define autoencoder net
        [ae_net, encoder_net, decoder_net] = self._create_autoencoder_net(ae_inputs, n_features, latent_dim) 
        print("AE net ")
        ae_net.summary()
        print("Encoder net ")
        encoder_net.summary()
        print("Decoder net ")
        decoder_net.summary()
            
        #Define adversarial net
        adv_net = self._create_adv_net(adv_inputs)
        
        #Turn on/off network weights
        self._trainable_ae_net = self._make_trainable(ae_net) 
        self._trainable_adv_net = self._make_trainable(adv_net) 
        
        #Compile models
        self._ae = self._compile_ae(ae_net) 
        self._encoder =  self._compile_encoder(encoder_net) 
        self._decoder =  self._compile_decoder(decoder_net) 
        self._ae_w_adv = self._compile_ae_w_adv(ae_inputs, ae_net, encoder_net, adv_net) 
        self._adv = self._compile_adv(ae_inputs, ae_net, encoder_net, adv_net)
        
        print("Autoencoder net with adv ")
        self._ae_w_adv.summary()
        
        #Define metrics
        self._val_metrics = None
        self._fairness_metrics = None
        
    #Freeze layers if network
    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable
       
    #Method for defining autoencoder network
    def _create_autoencoder_net(self, inputs, n_features, latent_dim):
        
        #Encoder
        dense1 = Dense(500, activation='relu')(inputs)
        dropout1 = Dropout(0.0)(dense1)
        latent_layer = Dense(latent_dim)(dropout1)
        
        #Decoder
        dense2 = Dense(500, activation='relu')
        dropout2 = Dropout(0.0)
        outputs = Dense(n_features)
        
        decoded = dense2(latent_layer)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)
        
        autoencoder = Model(inputs=[inputs], outputs=[decoded], name = 'autoencoder')
        encoder = Model(inputs=[inputs], outputs=[latent_layer], name = 'encoder')
        
        #Define decoder
        decoder_input = Input(shape=(latent_dim, )) 
        decoded = dense2(decoder_input)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)
        decoder = Model(inputs = decoder_input, outputs=[decoded],  name = 'decoder')
        
        return [autoencoder, encoder, decoder]
     
    #Method for defining adversarial network  
    def _create_adv_net(self, inputs):
        dense1 = Dense(50, activation='relu')(inputs)
        dense2 = Dense(50, activation='relu')(dense1)
        outputs = Dense(1, activation='sigmoid')(dense2)
        return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary')

    #Compile model
    def _compile_ae(self, ae_net):
        ae = ae_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
    
    #Compile modelModels)
    
    def _compile_encoder(self, encoder_net):
        ae = encoder_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
      
    #Compile model
    def _compile_decoder(self, decoder_net):
        ae = decoder_net
        self._trainable_ae_net(True) 
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
    
    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    #Compile autoencoder with adv loss
    #The model takes input features as input
    #Outputs classifier prediction + adversarial prediction from the classifier prediction
    def _compile_ae_w_adv(self, inputs, ae_net, encoder_net, adv_net):
        ae_w_adv = Model(inputs=[inputs], outputs = [ae_net(inputs)] + [adv_net(encoder_net(inputs))])
        self._trainable_ae_net(True) #classifier is trainable
        self._trainable_adv_net(False) #Freeze the adversary
        loss_weights = [1., -1 * self.lambda_val] #classifier loss - adversarial loss
        #Now compile the model with two losses and defined weights
        ae_w_adv.compile(loss=['mse', 'binary_crossentropy'], 
                          metrics=['mse', 'accuracy'], 
                          loss_weights=loss_weights,
                          optimizer='adam')
        return ae_w_adv

    #Compile adversarial model
    #Takes input features and outputs adversarial prediction
    def _compile_adv(self, inputs, ae_net, encoder_net, adv_net):
        adv = Model(inputs=[inputs], outputs=adv_net(encoder_net(inputs)))
        self._trainable_ae_net(False) #Freeze the classifier
        self._trainable_adv_net(True) #adversarial net is trainable
        adv.compile(loss=['binary_crossentropy'], 
                    metrics = ['accuracy'], optimizer='adam') 
        return adv
        
    #Pretrain all models
    def pretrain(self, x, z, validation_data=None, epochs=10):
        self._trainable_ae_net(True)
        self._ae.fit(x.values, x.values, epochs=epochs)
        self._trainable_ae_net(False)
        self._trainable_adv_net(True)
        
        if validation_data is not None:
            x_val, z_val = validation_data

        self._adv.fit(x.values, z.values,
                      validation_data = (x_val.values, z_val.values),
                        epochs=epochs, verbose=2)
        
    #Now do adversarial training
    def fit(self, x, z, validation_data=None, T_iter=250, batch_size=128):
        
        if validation_data is not None:
            x_val, z_val = validation_data

        self._val_metrics = pd.DataFrame()
        self._train_metrics = pd.DataFrame()
        
        #Go over all iterations
        for idx in range(T_iter):
            print("Iter ", idx)
            
            if validation_data is not None:
                
                #Predict with encoder
                x_pred = pd.DataFrame(self._ae.predict(x_val), index = x_val.index)
                self._val_metrics.loc[idx, 'MSE'] = mean_squared_error(x_val, x_pred)
                
            # train adversary
            self._trainable_ae_net(False)
            self._trainable_adv_net(True)
            print("Training adversary...")
            history = self._adv.fit(x.values, z.values, 
                                    validation_data = (x_val.values, z_val.values),
                                    batch_size=batch_size, epochs=1, verbose=1)
            self._train_metrics.loc[idx, 'Adversary accuracy'] = history.history['accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary accuracy'] = history.history['val_accuracy'][0]
            
            # train autoencoder
            self._trainable_ae_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            print("Training adversarial autoencoder...")
            history = self._ae_w_adv.fit(x.values[indices],
                                     [x.values[indices]] + [z.values[indices]],
                                     batch_size=batch_size, epochs=1, verbose=1,
                                     validation_data = (x_val.values,
                                     [x_val.values] + [z_val.values]))
            
            print("Autoencoder loss ",  history.history)
            keys = self._ae_w_adv.metrics_names
            
            #Record of interest results
            self._train_metrics.loc[idx, 'Total autoencoder loss'] = history.history['loss'][0]
            self._val_metrics.loc[idx, 'Total autoencoder loss'] = history.history['val_loss'][0]
             
            self._train_metrics.loc[idx, 'Autoencoder MSE'] = history.history['autoencoder_mse'][0]
            self._val_metrics.loc[idx, 'Autoencoder MSE'] = history.history['val_autoencoder_mse'][0]
                
            self._train_metrics.loc[idx, 'Adversary accuracy'] = history.history['adversary_accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary accuracy'] = history.history['val_adversary_accuracy'][0]
              
    
        #Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(60, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder training loss', lw = 5, color = '#27ae60')
        plt.plot(self._val_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder validation loss', lw = 5, color = '#f39c12')
        
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()
        
        #Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(60, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder training MSE', lw = 5, color = '#3498db')
        plt.plot(self._val_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder validation MSE', lw = 5, color = '#e74c3c')
           
        plt.plot(self._train_metrics['Adversary accuracy'], 
                 label = 'Adversary training accuracy', lw = 5, color = '#16a085')
        plt.plot(self._val_metrics['Adversary accuracy'], 
                 label = 'Adversary validation accuracy', lw = 5, color = '#9b59b6')
         
            
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()

        
import sys
run = int(sys.argv[1])
lambda_val = 0.1
latent_dim = 50

#Define model
adv_model = AdversarialDeconfoundingAutoencoder(n_features=X_train.shape[1], 
                                       latent_dim = latent_dim, lambda_val = lambda_val, random_seed = run)

#Pretrain both networks
adv_model.pretrain(X_train, Z_train, 
                    validation_data=(X_test, Z_test), epochs=5)

#Joint training
adv_model.fit(X_train, Z_train, 
        validation_data=(X_test, Z_test),
        T_iter = 3000)

#Generate embedding for all samples
embedding = adv_model._encoder.predict(X)
embedding_df = pd.DataFrame(embedding, index = X.index)
embedding_df.to_csv('ADV_FILES_SEX/ADV_Embedding_' + str(latent_dim) + 'L_lam' + str(lambda_val) + '_fold' + str(run) + '_3k_epochs.tsv')

#Record models
model_json = adv_model._encoder.to_json()
with open('ADV_FILES_SEX/ADV_encoder_' + str(latent_dim) + 'L_lam'+ str(lambda_val) +'_fold'+ str(run) +'_400_epochs.json', "w") as json_file:
    json_file.write(model_json)
adv_model._encoder.save_weights('ADV_FILES_SEX/ADV_encoder_' + str(latent_dim) + 'L_lam'+ str(lambda_val) +'_fold'+ str(run) +'_3k_epochs.h5')
print("Saved model to disk")

model_json = adv_model._decoder.to_json()
with open('ADV_FILES_SEX/ADV_decoder_' + str(latent_dim) + 'L_lam'+ str(lambda_val) +'_fold'+ str(run) +'_400_epochs.json', "w") as json_file:
    json_file.write(model_json)
adv_model._decoder.save_weights('ADV_FILES_SEX/ADV_decoder_' + str(latent_dim) + 'L_lam'+ str(lambda_val) +'_fold'+ str(run) +'_3k_epochs.h5')
print("Saved model to disk")

       
        
