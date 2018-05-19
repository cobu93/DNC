import numpy as np
import tensorflow as tf
from random import randint
from DatasetGenerator import DatasetGenerator


class CopyGenerator(DatasetGenerator):

	def generate(self, numVectors, numBits, batchSize = 1):		

		inputs = []
		outputs = []

		for i in range(0, batchSize):
			# Generate original input
			original = np.random.randint(2, size=(numVectors, numBits))

			input = np.copy(original)
			# Add zero in first space of each vector
			input = np.concatenate((input,  np.zeros(shape=(numVectors + 1, numBits), dtype=np.int_)), axis=0)	
			input = np.concatenate((input,  np.zeros(shape=(2 * numVectors + 1, 1), dtype=np.int_)), axis=1)
			
			# Add the flag and extra vector
			input[numVectors, numBits] = 1

			output = np.zeros(shape=(numVectors + 1, numBits), dtype=np.int_)
			output = np.concatenate((output, original), axis=0)


			inputs.append(input)
			outputs.append(output)

		
		return inputs, outputs


	def generate_train(self, vectors, bits, batch_size=1):
		return generate(vectors, bits, batch_size)


	def generate_test(self, vectors, bits, batch_size=1):
		return generate(vectors, bits, batch_size)

	def get_input_placeholder(self, vectors, bits):
		return tf.placeholder(tf.float32, [None, 2 * vectors  + 1, bits + 1])
	
	def get_output_placeholder(self, vectors, bits):
		return tf.placeholder(tf.float32, [None, 2 * vectors + 1, bits])

