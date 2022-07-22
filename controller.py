import data
import model
import tensorflow as tf

class Classic_GAN:
    
    #dataset directory
    
    
    #==== DATASET TOPOLOGY======
    # LENGTH = 64, BREATH = 64, DEPT = 3 (RGB)
    # PREFERABLY IN JPG FORMAT!
    #===========================
    directory = "./flower/"
    
    
    #Adam Neural Network Optimizer Parameters
    
    learning_rate = 0.003
    beta_1 = 0.3
    
    
    #Neural Network Training Parameters
    epochs = 20
    batch_size = 5
    
    
    #Number of cycles till plot
    plot_batch_size = 5
    
    G = model.GAN_model()
    D = data.get_dataset()
    
    def start_training(self):
        
        self.D.directory_location = self.directory
        self.G.learning_rate = self.learning_rate
        self.G.beta_1 = self.beta_1
        
        self.G.epochs = self.epochs
        self.G.batch_size = self.G.batch_size
        
        self.G.plot_batch_size = self.plot_batch_size
        
        
        self.G.load_dataset(self.D)
        self.G.generator_model()
        self.G.discriminator_model()
        self.G.GAN()
        self.G.GAN(tf.ones((1, 10)))
        self.G.GAN(self.G.get_random_input())
        self.G.training_sequences()

