import tensorflow as tf

class ContentAddressing():

	@staticmethod
	def address(key, intensity, memory, epsilon=1e-6):
		
		# Cosine similitude
		key_mem = tf.matmul(key, tf.transpose(memory)) # (KXW) (NXW)T = (KXW)(WXN) = KXN
		norm_memory_row = tf.sqrt( tf.reduce_sum( tf.multiply(memory, memory) , 1, keep_dims=True) ) # Nx1
		norm_key = tf.sqrt( tf.reduce_sum( tf.multiply(key, key) , 1, keep_dims=True) ) # KX1

		norms_sum = tf.multiply ( norm_key , tf.transpose(norm_memory_row)) + epsilon # KXN

		similitude = tf.divide(key_mem, norms_sum) # KXN

		return tf.nn.softmax(tf.multiply(tf.transpose(intensity), similitude)) #KXN





		
