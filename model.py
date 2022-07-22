import numpy as np
import pandas as pd
import tensorflow as tf
import data
import matplotlib.pyplot as plt
import time

class GAN_model:
    
    dataset = np.array([])
    generator = tf.keras.Sequential()
    discriminator = tf.keras.Sequential()
    GAN = tf.keras.Sequential()
    learning_rate = 0.001
    beta_1 = 0.3
    epochs = 5
    batch_size = 5
    plot_batch_size = 2
    
    
    def load_dataset(self, dataloader):
        dataloader.get_names_of_all_the_files()
        dataloader.load_database()
        self.dataset = np.array(dataloader.dataset.reshape(dataloader.instances, 64*64*3, -1)) /255
        return dataloader.dataset.reshape(dataloader.instances, 64*64*3, -1) /255
    
    def generator_model(self):
        self.generator.add(tf.keras.layers.Input(10))
        self.generator.add(tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='he_uniform'))
        self.generator.add(tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='he_uniform'))
        self.generator.add(tf.keras.layers.Dense(20, activation='relu', kernel_initializer='he_uniform'))
        self.generator.add(tf.keras.layers.Dense(self.dataset.shape[1], activation='sigmoid', kernel_initializer='he_uniform'))
        self.generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta_1), loss=tf.losses.BinaryCrossentropy(from_logits=True))
        return self.generator.summary()
    
    def discriminator_model(self):
        self.discriminator.add(tf.keras.layers.Input(self.dataset.shape[1]))
        self.discriminator.add(tf.keras.layers.Dense(20, activation='relu', kernel_initializer='he_uniform'))
        self.discriminator.add(tf.keras.layers.Dense(20, activation='linear', kernel_initializer='he_uniform'))
        self.discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta_1), loss=tf.losses.BinaryCrossentropy(from_logits=True))
        return self.discriminator.summary()
    
    def GAN(self):
        self.GAN = tf.keras.Sequential([self.generator, self.discriminator])
        self.GAN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta_1), loss=tf.losses.BinaryCrossentropy(from_logits=True))
        
    
    
    #Training Steps and Sequences here!
    
    def get_random_input(self):
        return np.random.rand(1, 10)
    
    def training_sequences(self):
        self.GAN(tf.ones((1, 10))) #Test!
        for epoch in range(self.epochs):
            self.GAN.layers[0].trainable = False
            self.GAN.layers[1].trainable = True
            for batch in range(self.batch_size):
                # 0 - fake data, 1 - real data
                self.GAN.fit(self.get_random_input(), np.array([0]))
            
            self.GAN.layers[1].fit(self.dataset[epoch % self.dataset.shape[0]].reshape(1, -1), np.array([1]))
            
            self.GAN.layers[0].trainable = True
            self.GAN.layers[1].trainable = False
            
            self.GAN.fit(self.get_random_input(), np.array([1]))
            
            print("Epoch: " + str(epoch) + " of " + str(self.epochs) + " complete.")
            if epoch % self.plot_batch_size == 0:
                print("plotting data!")
                plotdata = np.array(self.GAN.layers[0](self.get_random_input()))
                plotdata = plotdata.reshape(-1, 1).reshape(64, 64, 3)
                plt.imshow(plotdata)
                plt.show()
                time.sleep(1)