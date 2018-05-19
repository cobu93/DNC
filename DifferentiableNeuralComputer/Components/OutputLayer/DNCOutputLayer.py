
import tensorflow as tf
from DifferentiableNeuralComputer.Components.General.FeedForwardNN import FeedForwardNN
from DifferentiableNeuralComputer.Components.General.Utility import Utility
from DifferentiableNeuralComputer.Components.General.Layer import OutputLayerDescription

class DNCOutputLayer():

   def __init__(self, layers_desc, name='linear_output'):
      self.layers_desc = layers_desc
      self.ffnn = FeedForwardNN(layers_desc, name)

      '''
      input:
         layers_desc = Description of each layer (LayerDescription object type )
         error_func = Error function, must has this input:output, expected_output 
         opt_func = Optimization function. Must have next input: error
      '''

   def run_output_layer(self, inputs):
      if len(inputs) != 2:
         raise ValueError('For DNC output layer inputs must be of length 2. Header and read vectors.')

      output_vector = self.ffnn.run_feed_forward_nn(inputs[0])

      output_layer = None

      for layer in self.layers_desc:

         if layer.__class__ is OutputLayerDescription:
            output_layer = layer
            break

      if not output_layer:
         raise Exception('Not OutputLayerDescription was found in layer description.')

      return tf.add(output_vector, Utility._linear(inputs[1], output_layer.size, output_layer.has_bias))

				


