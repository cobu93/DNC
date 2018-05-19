import tensorflow as tf
import numpy as np
import math

class TemporalLinkage():

	@staticmethod
	# address(Scalar, 1XN, Scalar, 1XN, 1XN)			content_weights
	def link(w_weights, last_precedence, last_linkage_matrix, last_r_weights):
		precedence_vector = TemporalLinkage.precedence(w_weights, last_precedence)#1XN
		link_matrix = TemporalLinkage.linkage_matrix(w_weights, last_precedence, last_linkage_matrix)
		backward_weights = TemporalLinkage.backward_weighting(last_r_weights, link_matrix)
		forward_weights = TemporalLinkage.forward_weighting(last_r_weights, link_matrix)
		
		return precedence_vector, link_matrix, forward_weights, backward_weights

	@staticmethod
	def precedence(w_weights, last_precedence): #1 X N
		return tf.add((1 - tf.reduce_sum(w_weights)) * last_precedence, w_weights) 


	@staticmethod
	def linkage_matrix(w_weights, last_precedence, last_linkage_matrix): #N X N
		rows = tf.size(w_weights)
		w_weights_tiled = tf.tile(tf.reshape(w_weights, [rows]), [rows])
		w_weights_tiled = tf.reshape(w_weights_tiled, [rows, rows])
		
		l_matrix = tf.multiply(1 - w_weights_tiled - tf.transpose(w_weights_tiled), last_linkage_matrix) + tf.tensordot(tf.transpose(w_weights), last_precedence, axes=1)

		l_matrix = tf.matrix_set_diag(l_matrix, tf.zeros([rows]))

		return l_matrix

	@staticmethod
	def forward_weighting(last_r_weights, linkage_matrix): #1 X N
		forward_weights = tf.transpose(tf.matmul(linkage_matrix, tf.transpose(last_r_weights)))
		return forward_weights

	@staticmethod
	def backward_weighting(last_r_weights, linkage_matrix): #1 X N
		backward_weights = tf.transpose(tf.matmul(tf.transpose(linkage_matrix), tf.transpose(last_r_weights)))
		return backward_weights




