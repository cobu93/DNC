from DifferentiableNeuralComputer.Components.MemoryAccess.ContentManagment.ContentAddressing import ContentAddressing
from DifferentiableNeuralComputer.Components.MemoryAccess.ContentManagment.DynamicAllocation import DynamicAllocation
import tensorflow as tf

class Writer():


	def run_writer(self, memory, w_weights, w_key, w_intensity, r_weights, r_free_gates, usage_vector, allocation_gate, write_gate, erase_vector, add_vector, epsilon=1e-6):

		content_vector = ContentAddressing.address(
			key=w_key, 
			intensity=w_intensity, 
			memory=memory,
			epsilon=epsilon
			) #1XN


		allocation_weights, usage_vec = DynamicAllocation.allocate(
			free_gates=r_free_gates, 
			last_read_weights=r_weights, 
			last_usage_vector=usage_vector, 
			last_write_weights=w_weights
			) # 1XN

		write_weights = tf.multiply(
							write_gate, 
							tf.multiply(allocation_gate, allocation_weights) + tf.multiply(1 - allocation_gate, content_vector)
						)

		n_memory = tf.multiply(
    			memory, 
    			tf.ones(tf.shape(memory)) - tf.matmul(tf.transpose(write_weights), erase_vector)
    		) + tf.matmul(tf.transpose(write_weights), add_vector)

	
		return write_weights, usage_vec, n_memory
