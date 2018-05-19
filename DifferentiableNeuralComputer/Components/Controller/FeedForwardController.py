import tensorflow as tf
from DifferentiableNeuralComputer.Components.General.FeedForwardNN import FeedForwardNN

class FeedForwardController():

	def __init__(self, layers_desc, name='ffnn_controller'):
		self.ffnn = FeedForwardNN(layers_desc, name)

   	'''
   	input:
   		layers_desc = Description of each layer (LayerDescription object type )
   		error_func = Error function, must has this input:output, expected_output 
   		opt_func = Optimization function. Must have next input: error
   	'''
	def run_controller(self, inputs):
		return self.ffnn.run_feed_forward_nn(inputs)
		
		


