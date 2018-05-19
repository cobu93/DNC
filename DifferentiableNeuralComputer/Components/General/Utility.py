import tensorflow as tf
from tensorflow.python.util import nest

class Utility():


	_BIAS_VARIABLE_NAME = "bias"
	_WEIGHTS_VARIABLE_NAME = "weights"

	@staticmethod
	def _linear(args, output_size, bias, bias_initializer=None, kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5)):

		"""
			Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
			Args:
			args: a 2D Tensor or a list of 2D, batch x n, Tensors.
			output_size: int, second dimension of W[i].
			bias: boolean, whether to add a bias term or not.
			bias_initializer: starting value to initialize the bias
			(default is all zeros).
			kernel_initializer: starting value to initialize the weight.
			Returns:
			A 2D Tensor with shape [batch x output_size] equal to
			sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
			Raises:
			ValueError: if some of the arguments has unspecified or wrong shape.
		"""

		if args is None or (nest.is_sequence(args) and not args):
			raise ValueError("`args` must be specified")
		
		if not nest.is_sequence(args):
			args = [args]

		# Calculate the total size of arguments on dimension 1.
		total_arg_size = 0

		shapes = [a.get_shape() for a in args]
		for shape in shapes:
			if shape.ndims != 2:
				raise ValueError("linear is expecting 2D arguments: %s" % shapes)
		
			if shape[1].value is None:
				raise ValueError("linear expects shape[1] to be provided for shape %s, "
						"but saw %s" % (shape, shape[1]))
			else:
				total_arg_size += shape[1].value

		dtype = [a.dtype for a in args][0]

		# Now the computation.
		scope = tf.get_variable_scope()

		with tf.variable_scope(scope) as outer_scope:
			weights = tf.get_variable(
				Utility._WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
				dtype=dtype,
				initializer=kernel_initializer)
			if len(args) == 1:
				res = tf.matmul(args[0], weights)
			else:
				res = tf.matmul(tf.concat(args, 1), weights)
				
			if not bias:
				return res
			
			with tf.variable_scope(outer_scope) as inner_scope:
				inner_scope.set_partitioner(None)
				if bias_initializer is None:
					bias_initializer = tf.constant_initializer(1e-8, dtype=dtype)
			
				biases = tf.get_variable(
						Utility._BIAS_VARIABLE_NAME, [output_size],
						dtype=dtype,
						initializer=bias_initializer)

		return tf.nn.bias_add(res, biases)


staticmethod
def _concat(args):

	"""
		Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
		Args:
		args: a 2D Tensor or a list of 2D, batch x n, Tensors.
		output_size: int, second dimension of W[i].
		bias: boolean, whether to add a bias term or not.
		bias_initializer: starting value to initialize the bias
		(default is all zeros).
		kernel_initializer: starting value to initialize the weight.
		Returns:
		A 2D Tensor with shape [batch x output_size] equal to
		sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
		Raises:
		ValueError: if some of the arguments has unspecified or wrong shape.
	"""

	if args is None or (nest.is_sequence(args) and not args):
		raise ValueError("`args` must be specified")
	
	if not nest.is_sequence(args):
		args = [args]

	shapes = [a.get_shape() for a in args]
	for shape in shapes:
		if shape.ndims != 2:
			raise ValueError("linear is expecting 2D arguments: %s" % shapes)
	
		if shape[1].value is None:
			raise ValueError("linear expects shape[1] to be provided for shape %s, "
					"but saw %s" % (shape, shape[1]))
		

	return tf.concat(args, 1)
			