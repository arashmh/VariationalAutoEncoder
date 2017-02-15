from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from config import latent_dim

class SamplingLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SamplingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SamplingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        z_mean, z_log_var = x
        dynamicBatchSize = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)