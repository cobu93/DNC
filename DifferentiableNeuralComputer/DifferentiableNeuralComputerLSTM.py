"""
Description: A general structure of Differentiable Neural Computer
Author: Uriel Corona Bermudez

"""

from Components.General.Layer import *
from Components.Controller.FeedForwardController import FeedForwardController
from Components.Controller.LSTMController import LSTMController
from Components.OutputLayer.DNCOutputLayer import DNCOutputLayer
from Components.MemoryAccess.Parameters.InterfaceParameters import InterfaceParameters

from Components.MemoryAccess.Read.Reader import Reader
from Components.MemoryAccess.Write.Writer import Writer

import tensorflow as tf
from tensorflow.python.util import nest



class DifferentiableNeuralComputer():


	SCALAR_SIZE = 1

	def __init__(self, osize, hsize, mrows, mcolumns, rvectors, epsilon=1e-6):

		self.osize = osize
		self.hsize = hsize
		self.mrows = mrows
		self.mcolumns = mcolumns
		self.rvectors = rvectors
		self.epsilon = epsilon

		
		
		self.controller = LSTMController(
			layers_desc = [
				InputLayerDescription('input_header'), 
				OutputLayerDescription(self.hsize, 'header')
			],
			name='controller'
			)
		
		self.output_layer = DNCOutputLayer(
			layers_desc = [
				InputLayerDescription('input_y'), 
				OutputLayerDescription(self.osize, 'output', tf.nn.tanh, has_bias=False)
			],
			name='dnc_output'
			)

		self.interface_parameters = InterfaceParameters(
			key_w_layers_desc=[
				InputLayerDescription('input_key'), 
				OutputLayerDescription(self.mcolumns, 'write_key', tf.nn.tanh)
			], 

			intensity_w_layers_desc=[
				InputLayerDescription('input_intensity'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'write_intensity', tf.nn.softplus)
			],  

			erase_layers_desc=[
				InputLayerDescription('input_erase'), 
				OutputLayerDescription(self.mcolumns, 'erase', tf.nn.sigmoid)
			], 

			add_layers_desc=[
				InputLayerDescription('input_add'), 
				OutputLayerDescription(self.mcolumns, 'add', tf.nn.tanh)
			],

			allocation_layers_desc=[
				InputLayerDescription('input_allocation'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'allocation', tf.nn.sigmoid)
			], 

			write_gate_layers_desc=[
				InputLayerDescription('input_write_gate'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'write_gate', tf.nn.sigmoid)
			],

			free_gate_layers_desc=[
				InputLayerDescription('input_free_gates'), 
				OutputLayerDescription(self.SCALAR_SIZE  * self.rvectors, 'free_gates', tf.nn.sigmoid)
			],

			key_r_layers_desc=[
				InputLayerDescription('input_key'), 
				OutputLayerDescription(self.mcolumns * self.rvectors, 'read_keys', tf.nn.tanh)
			], 

			intensity_r_layers_desc=[
				InputLayerDescription('input_intensity'), 
				OutputLayerDescription(self.SCALAR_SIZE  * self.rvectors, 'read_intensity', lambda x: 1 + tf.log(1 + tf.exp(x)))
			],

			read_mode_layers_desc=[
				InputLayerDescription('input_read_mode'), 
				OutputLayerDescription(3  * self.rvectors, 'read_mode', tf.nn.softmax)
			],

			name='interface_parameters'

		)

		self.reader = Reader()
		self.writer = Writer()



	def run_dnc(self, inputs, state):

		print('running...')
		header, memory, read_vecs, write_weights, read_weights, usage_vector, precedence_vector, linkage_matrix = state



		memory = tf.reshape(memory, [self.mrows, self.mcolumns], 'reshape_memory')
		read_weights = tf.reshape(read_weights, [self.rvectors, self.mrows], 'reshape_read_weights')
		linkage_matrix = tf.reshape(linkage_matrix, [self.mrows, self.mrows], 'reshape_linkage_matrix')
		

		n_header = self.controller.run_controller([header, inputs, read_vecs])
		y = self.output_layer.run_output_layer([n_header, read_vecs])

		

		w_key, w_intensity, w_add, w_erase, w_alloc, w_gate, r_free_gates, r_keys, r_intensities, r_modes = self.interface_parameters.run_generation([n_header])
		r_modes = tf.reshape(r_modes, [self.rvectors, 3])
		r_keys = tf.reshape(r_keys, [self.rvectors, self.mcolumns])

		n_write_weights, n_usage_vector, n_memory = self.writer.run_writer(memory, write_weights, w_key, w_intensity, read_weights, r_free_gates, usage_vector, w_alloc, w_gate, w_erase, w_add, self.epsilon)
		n_read_weights, n_read_vecs, n_precedence_vector, n_linkage_matrix = self.reader.run_reader(memory, read_weights, r_keys, r_intensities, r_modes, n_write_weights, precedence_vector, linkage_matrix, self.epsilon)

				

		n_memory = tf.reshape(n_memory, [1, self.mrows * self.mcolumns])
		n_read_weights = tf.reshape(n_read_weights, [1, self.rvectors * self.mrows])
		n_linkage_matrix = tf.reshape(n_linkage_matrix, [1, self.mrows * self.mrows])
		n_read_vecs = tf.reshape(n_read_vecs, [1, self.rvectors * self.mcolumns])
		
		
		return y, (n_header, n_memory, n_read_vecs, n_write_weights, n_read_weights, n_usage_vector, n_precedence_vector, n_linkage_matrix)




class DNCCell(tf.contrib.rnn.RNNCell):
 

	def __init__(self, osize, hsize, mrows, mcolumns, rvectors, epsilon=1e-6):
		tf.contrib.rnn.RNNCell.__init__(self)
		self.dnc = DifferentiableNeuralComputer(osize=osize, hsize=hsize, mrows=mrows, mcolumns=mcolumns, rvectors=rvectors, epsilon=epsilon)

       
	@property
	def state_size(self):
		return (self.dnc.hsize, self.dnc.mrows * self.dnc.mcolumns, self.dnc.rvectors * self.dnc.mcolumns, self.dnc.mrows, self.dnc.rvectors * self.dnc.mrows, self.dnc.mrows, self.dnc.mrows, self.dnc.mrows * self.dnc.mrows)
            
	@property
	def output_size(self):
		return self.dnc.osize


	def __call__(self, inputs, state):
		return self.dnc.run_dnc(inputs, state)
		