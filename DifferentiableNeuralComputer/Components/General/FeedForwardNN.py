
import tensorflow as tf
from Utility import Utility


class FeedForwardNN():

	def __init__(self, layers_desc, scope):
		self.layers = layers_desc
		self.scope = scope
		self.h = {}

	def run_feed_forward_nn(self, inputs):
		
		self.h.clear()

		with tf.variable_scope(self.scope):

			for i in range(0, len(self.layers)):

				if self.layers[i].is_input:
					self.h[i] = inputs #if isinstance(inputs, list) else [inputs]
									
				else:
					with tf.variable_scope(self.layers[i].name):

						self.h[i] =  self.layers[i].activation(
										Utility._linear(
											self.h[i - 1],#(self.h[i - 1] if isinstance(self.h[i - 1], list) else [self.h[i - 1]]), 
											self.layers[i].size, 
											self.layers[i].has_bias
										)
									) 

				
		return self.h[len(self.layers) - 1]
		
	