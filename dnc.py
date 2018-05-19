#!/usr/bin/python

from DifferentiableNeuralComputer.DifferentiableNeuralComputer import DifferentiableNeuralComputer
from DifferentiableNeuralComputer.DifferentiableNeuralComputer import DNCCell

import tensorflow as tf
import argparse, os, sys, inspect


def build_model(i_placeholder, o_placeholder, header_size, output_size, memory_rows, memory_columns, learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10):


	HEADER_SIZE = header_size
	OUTPUT_SIZE = output_size

	MEMORY_ROWS = memory_rows
	MEMORY_COLUMNS = memory_columns

	LEARNING_RATE = learning_rate
	DECAY = decay
	MOMENTUM = momentum

	NUM_READ_VECTORS=5

	EPSILON = epsilon

	global_step = tf.Variable(0, name='global_step', trainable=False)


	dnc = DNCCell(
		osize=OUTPUT_SIZE, 
		hsize=HEADER_SIZE, 
		mrows=MEMORY_ROWS, 
		mcolumns=MEMORY_COLUMNS,
		rvectors=NUM_READ_VECTORS,
		epsilon=EPSILON
		)


	# End of associative recall config
 	# header, memory, read_vecs, write_weights, read_weights, usage_vector, precedence_vector, linkage_matrix = state

	init_state = (
		tf.zeros([1, HEADER_SIZE], name='init_read_vector'),
		tf.zeros([1, MEMORY_ROWS * MEMORY_COLUMNS], name='init_memory'),
		tf.zeros([1, NUM_READ_VECTORS * MEMORY_COLUMNS], name='init_read_vectors'),
		tf.truncated_normal([1, MEMORY_ROWS], stddev=0.2, mean=0.5, name='init_write_weights'), # Must be initialized with positive values
		tf.truncated_normal([1, NUM_READ_VECTORS * MEMORY_ROWS], stddev=0.2, mean=0.5, name='init_read_weights'), # Must be initialized with positive values
		tf.zeros([1, MEMORY_ROWS], name='init_usage_vector'),
		tf.zeros([1, MEMORY_ROWS], name='init_precedence_vector'),
		tf.zeros([1, MEMORY_ROWS * MEMORY_ROWS], name='init_linkage_matrix'),
		)

	
	output, _ = tf.nn.dynamic_rnn(dnc, i_placeholder, dtype=tf.float32, initial_state=init_state)

	# cross_entropy = tf.multiply(o_placeholder, tf.log(output + EPSILON)) + tf.multiply(1 - o_placeholder, tf.log(1 - output + EPSILON))
	# cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'cross_entropy', summarize=20)
	
	mse = tf.multiply(o_placeholder - output, o_placeholder - output)
	# mse = tf.Print(mse, [mse], 'mse', summarize=20)
	error = mse

	loss = tf.reduce_mean(error)
	# loss = -tf.reduce_mean(cross_entropy)
	# loss = tf.Print(loss, [loss], 'loss', summarize=20)

	optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM)

	training_op = optimizer.minimize(loss, global_step=global_step)

	prediction_test = tf.round(output)



	return output, error, loss, training_op, prediction_test, global_step









def train(i_placeholder, o_placeholder, num_vectors, num_bits, training_op, output, prediction_test, error, folder, global_step, gen_function, batch_size=1, epochs=20000):

	NUM_VECTORS = num_vectors
	NUM_BITS = num_bits
	EPOCHS = epochs
	BATCH_SIZE = batch_size
	# Start of associative recall config
	
	#Create folder if not exists
	if not os.path.exists(folder):
		os.makedirs(folder)
		

	restore_path = tf.train.latest_checkpoint(folder)

	if(restore_path):
		print('Restoring model from %s.' % restore_path)
		saver = tf.train.Saver()
		saver.restore(sess, restore_path)
		print('Model restored.')
	else:
		init = tf.global_variables_initializer()	
		saver = tf.train.Saver()
		init.run()

	for epoch in range(1, EPOCHS):
		x_batch, y_batch = gen_function.generate(NUM_VECTORS, NUM_BITS, BATCH_SIZE)
		_, out, prediction = sess.run([training_op, output, prediction_test], feed_dict={i_placeholder: x_batch, o_placeholder: y_batch})
		if (epoch % 100) == 0:
			mse = error.eval(feed_dict={i_placeholder: x_batch, o_placeholder: y_batch})
			# train_writer.add_summary(summary, epoch)
			#out = state[1].eval(feed_dict={X: x_batch, Y: y_batch})

			print('\n---------------------- Epoch %d ----------------------' % epoch)
			print('MSE:' + str(mse))

			print('\nInput: ')
			print(x_batch)
			print('\nNTM output: ')
			print(out)
			print('\nNTM output (rounded): ')
			print(prediction)
			print('\nTarget output: ')
			print(y_batch)

			save_path = saver.save(sess, folder + '/dnc', global_step=global_step)
			print('\nModel saved in file: %s' % save_path)

			
			# out = outputs.eval(feed_dict={X: x_batch, Y: y_batch})
			# print('y:')
			# print(out)

			# h = state[0].eval(feed_dict={X: x_batch, Y: y_batch})
			# print('head:')
			# print(h)


			# mem = state[1].eval(feed_dict={X: x_batch, Y: y_batch})
			# print('mem:')
			# print(mem)

			# rv = state[2].eval(feed_dict={X: x_batch, Y: y_batch})
			# print('read_vec:')
			# print(rv)

			# ww = state[3].eval(feed_dict={X: x_batch, Y: y_batch})
			# print('write_w:')
			# print(ww)

			# rw = state[4].eval(feed_dict={X: x_batch, Y: y_batch})
			# print('read_w:')
			# print(rw)




			#print(epoch, "\toutput:", out)

	# train_writer.flush()
	# train_writer.close()


def test(i_placeholder, o_placeholder, prediction_test, num_vectors, num_bits, folder, gen_function, batch_size=1, num_tests=10):

	NUM_VECTORS = num_vectors
	NUM_BITS = num_bits
	NUM_TESTS=num_tests
	BATCH_SIZE = batch_size


	restore_path = tf.train.latest_checkpoint(folder)
	print('Restoring model from %s.' % restore_path)
	saver = tf.train.Saver()
	saver.restore(sess, restore_path)
	print('Model restored.')

	for i in range(0, NUM_TESTS):
		x_test, y_test = gen_function.generate(NUM_VECTORS, NUM_BITS, BATCH_SIZE)
		prediction = sess.run(prediction_test, feed_dict={i_placeholder: x_test, o_placeholder: y_test})


		print('\n---------------------- Test %d ----------------------' % i)
		print('\nInput: ')
		print(x_test)
		print('\nNTM output (rounded): ')
		print(prediction)
		print('\nTarget output: ')
		print(y_test)


















parser = argparse.ArgumentParser(description='A Differentiable Neural Computer.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store', dest='epochs', type=int, help='Use for train the DNC.')
group.add_argument('--info', action='store_true', help='Use for get information about DNC.')
group.add_argument('--test', action='store', dest='examples', type=int, help='Use for test the DNC.')


parser.add_argument('--task_module', action='store', dest='task_module', help='Module for lookup task dataset generator.', required=True)

parser.add_argument('--task_class', action='store', dest='task_class', help='Class that will feed test and training information.', required=True)


group_tb = parser.add_mutually_exclusive_group(required=False)
group_tb.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Activates tensorboard graphics.')
group_tb.add_argument('--no-tensorboard', dest='tensorboard', action='store_false', help='Deactivates tensorboard graphics. Default.')

parser.add_argument('--folder', action='store', dest='folder', default='', help='Destination folder to save/recover test/train models.')
parser.add_argument('--vectors', action='store', dest='vectors', default=5, type=int, help='Vectors number to use in task.')
parser.add_argument('--bits', action='store', dest='bits', default=5, type=int, help='Bits on each vector used in task.')

parser.add_argument('--mem_rows', action='store', dest='rows', default=128, type=int, help='Rows in memory.')
parser.add_argument('--mem_cols', action='store', dest='cols', default=50, type=int, help='Columns in memory.')


parser.add_argument('--l_rate', action='store', dest='learning_rate', default=8e-5, type=float, help='Learning rate.')
parser.add_argument('--decay', action='store', dest='decay', default=0.7, type=float, help='Decay.')
parser.add_argument('--momentum', action='store', dest='momentum', default=0.7, type=float, help='Momentum.')


parser.set_defaults(tensorboard=False)


args = parser.parse_args()



NUM_VECTORS = args.vectors
NUM_BITS = args.bits

HEADER_SIZE = 100
OUTPUT_SIZE = NUM_BITS

MEMORY_ROWS = args.rows # N
MEMORY_COLUMNS = args.cols # M

LEARNING_RATE = args.learning_rate
DECAY = args.decay
MOMENTUM = args.momentum


module = __import__(args.task_module, fromlist=[args.task_class])
TASK_GENERATOR = getattr(module, args.task_class)
task_generator = TASK_GENERATOR()


sess = tf.InteractiveSession()

X = task_generator.get_input_placeholder(NUM_VECTORS, NUM_BITS)
Y = task_generator.get_output_placeholder(NUM_VECTORS, NUM_BITS)


output, cross_entropy, loss, training_op, prediction_test, global_step = build_model(
																		i_placeholder = X,
																		o_placeholder = Y,
																		header_size=HEADER_SIZE, 
																		output_size=OUTPUT_SIZE, 
																		memory_rows=MEMORY_ROWS, 
																		memory_columns=MEMORY_COLUMNS, 
																		learning_rate=LEARNING_RATE, 
																		decay=DECAY, 
																		momentum=MOMENTUM
																		)



if args.epochs:
	train(i_placeholder=X, 
			o_placeholder=Y, 
			num_vectors=NUM_VECTORS, 
			num_bits=NUM_BITS, 
			training_op=training_op, 
			output=output, 
			prediction_test=prediction_test,
			error=loss, 
			epochs=args.epochs,
			folder='Parameters/' + args.folder,
			global_step=global_step,
			gen_function=task_generator
			)

elif args.examples:
	test(i_placeholder=X, 
			o_placeholder=Y, 
			prediction_test=prediction_test, 
			num_vectors=NUM_VECTORS, 
			num_bits=NUM_BITS, 
			num_tests=args.examples,
			folder='Parameters/' + args.folder,
			gen_function=task_generator)
	
