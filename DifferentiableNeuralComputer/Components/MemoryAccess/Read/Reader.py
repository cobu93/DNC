from DifferentiableNeuralComputer.Components.MemoryAccess.ContentManagment.ContentAddressing import ContentAddressing
from DifferentiableNeuralComputer.Components.MemoryAccess.ContentManagment.TemporalLinkage import TemporalLinkage
import tensorflow as tf

class Reader():


	#def run_reader(self, memory, r_weights, r_key, r_shift, r_intensity, r_interpolation, r_sharpen, epsilon=1e-6):
	#																							RX3
	def run_reader(self, memory, r_weights, r_keys, r_intensities, read_mode_vector, n_w_weights, precedence_vector, linkage_matrix, epsilon=1e-6):

		content_vector = ContentAddressing.address(
			key=r_keys, 
			intensity=r_intensities, 
			memory=memory,
			epsilon=epsilon
			)

		n_precedence, n_linkage_matrix, forward_weights, backward_weights = TemporalLinkage.link(
			w_weights=n_w_weights,
			last_precedence=precedence_vector,
			last_linkage_matrix=linkage_matrix,
			last_r_weights=r_weights
			) #1XN

		r_weights = tf.multiply(tf.transpose([read_mode_vector[:, 0]]), backward_weights) + tf.multiply(tf.transpose([read_mode_vector[:, 1]]), content_vector) + tf.multiply(tf.transpose([read_mode_vector[:, 2]]), forward_weights)

		r_vecs = tf.transpose(tf.matmul(tf.transpose(memory), tf.transpose(r_weights)))
		

		return r_weights, r_vecs, n_precedence, n_linkage_matrix
