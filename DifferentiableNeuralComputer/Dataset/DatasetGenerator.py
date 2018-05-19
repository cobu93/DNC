import abc

class DatasetGenerator(object):
	__metaclass__ = abc.ABCMeta
	
	@abc.abstractmethod
	def generate_train(self, vectors, bits, batch_size):
		raise NotImplementedError('Users must define generate_train to use this base class')


	@abc.abstractmethod
	def generate_test(self, vectors, bits, batch_size):
		raise NotImplementedError('Users must define generate_test to use this base class')


	@abc.abstractmethod
	def get_input_placeholder(self, vectors, bits):
		raise NotImplementedError('Users must define get_input_placeholder to use this base class')
	
	@abc.abstractmethod
	def get_output_placeholder(self, vectors, bits):
		raise NotImplementedError('Users must define get_output_placeholder to use this base class')