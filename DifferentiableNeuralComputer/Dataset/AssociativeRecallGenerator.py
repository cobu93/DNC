import numpy as np
import tensorflow as tf
from random import randint
from DatasetGenerator import DatasetGenerator

class AssociativeRecallGenerator(DatasetGenerator):

	def generate(self, numVectors, numBits, batchSize):
		inputs = []
		outputs = []

		for i in range(0, batchSize):
			# Generate original input
			original = np.random.randint(2, size=(numVectors, numVectors))
			position = randint(1, numVectors)
			original = np.concatenate((original,  np.zeros(shape=(1, numVectors), dtype=np.int_)), axis=0)
			original[numVectors, numVectors - position] = 1
		
			input = np.copy(original)
			input = np.concatenate((input,  np.zeros(shape=(1, numVectors), dtype=np.int_)), axis=0)
			input = np.concatenate((input,  np.zeros(shape=(numVectors + 2, 1), dtype=np.int_)), axis=1)
			input[numVectors, numVectors] = 1
		
			# Add the flag and extra vector
			output = np.zeros(shape=(numVectors + 1, numVectors), dtype=np.int_)
			vector = original[numVectors - position]
			output = np.concatenate((output, [vector]), axis=0)
		
			inputs.append(input)
			outputs.append(output)
		
		return inputs, outputs


	def generate_train(self, vectors, bits, batch_size=1):
		return generate(vectors, bits, batch_size)


	def generate_test(self, vectors, bits, batch_size=1):
		return generate(vectors, bits, batch_size)

	def get_input_placeholder(self, vectors, bits):
		return tf.placeholder(tf.float32, [None, vectors  + 2, bits + 1])
	
	def get_output_placeholder(self, vectors, bits):
		return tf.placeholder(tf.float32, [None, vectors + 2, bits])