import tensorflow as tf
from DifferentiableNeuralComputer.Components.General.Utility import Utility

class LSTMController():

	def __init__(self, layers_desc, name='lstm_controller'):
		#lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
		self.layers = layers_desc
		self.scope = name
		self.h = {}

   	'''
   	input:
   		layers_desc = Description of each layer (LayerDescription object type )
   		error_func = Error function, must has this input:output, expected_output 
   		opt_func = Optimization function. Must have next input: error
   	'''
	def run_controller(self, inputs):
	
		self.h.clear()
		def_layers = []
		length = 0 
		output_size = 0

		with tf.variable_scope(self.scope):



			for i in range(0, len(self.layers)):

				if self.layers[i].is_input:
					# self.h[i] = Utility._concat(inputs)
					# self.h[i] = Utility._concat([self.h[i], tf.zeros([])])
					self.h[i] = tf.concat(inputs, -1) #if isinstance(inputs, list) else [inputs]
					length = self.h[i].get_shape().as_list()[1]
									
				else:
					# tf.size(self.h[i - 1])
					lstm_cell = tf.contrib.rnn.LSTMCell(num_units=length)

					# outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, self.h[i - 1], dtype=tf.float32)
					outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, tf.reshape(self.h[i - 1], [1, 1, length]), dtype=tf.float32)

					self.h[i] = outputs

				if self.layers[i].is_output:					
					output_size = self.layers[i].size

		return tf.reshape(self.h[len(self.layers) - 1], [1, length])[:, (length - output_size): length]



