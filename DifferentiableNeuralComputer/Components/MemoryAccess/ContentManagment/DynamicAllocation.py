import tensorflow as tf
import numpy as np
import math

class DynamicAllocation():

	@staticmethod
	# address(Scalar, 1XN, Scalar, 1XN, 1XN)			content_weights
	def allocate(free_gates, last_read_weights, last_usage_vector, last_write_weights):
		retention_vector = DynamicAllocation.retention(free_gates, last_read_weights)#1XN
		usage_vector = DynamicAllocation.usage(retention_vector, last_usage_vector, last_write_weights)#1XN
		free_list = DynamicAllocation.free_list_sorted(usage_vector)#1XN
		allocation_vector = DynamicAllocation.allocation(usage_vector, free_list)

		return allocation_vector, usage_vector # 1 X N

	@staticmethod
	def retention(free_gates, last_read_weights): #1 X N
		return tf.transpose(tf.reduce_prod(1 - tf.multiply(free_gates, tf.transpose(last_read_weights)), axis=1))


	@staticmethod
	def usage(retention_vector, last_usage_vector, last_write_weights): #1XN
		return tf.multiply(last_usage_vector + last_write_weights - tf.multiply(last_usage_vector, last_write_weights), retention_vector)


	@staticmethod
	def free_list_sorted(usage_vector):#1XN
		_, indices = tf.nn.top_k(usage_vector, k=tf.size(usage_vector))
		return tf.reverse(indices, axis=[-1])

	@staticmethod
	def allocation(usage_vector, free_list):#1XN
		# allocation_weights = tf.zeros([1, tf.size(free_list)])

		u_vector = tf.reshape(usage_vector, [tf.size(usage_vector)])
		f_list = tf.reshape(free_list, [tf.size(free_list)])



		allocation_weights = tf.gather(tf.multiply(1 - tf.gather(u_vector, f_list), tf.cumprod(tf.gather(u_vector, f_list))), f_list)

		return allocation_weights


