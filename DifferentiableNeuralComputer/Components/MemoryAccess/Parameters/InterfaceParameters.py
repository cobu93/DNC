from DifferentiableNeuralComputer.Components.General.FeedForwardNN import FeedForwardNN

class InterfaceParameters():

	def __init__(self, key_w_layers_desc, intensity_w_layers_desc, erase_layers_desc, add_layers_desc, allocation_layers_desc, write_gate_layers_desc, free_gate_layers_desc, key_r_layers_desc, intensity_r_layers_desc, read_mode_layers_desc, name='interface_parameters'):
		self.key_w_nn = FeedForwardNN(key_w_layers_desc, name)
		self.intensity_w_nn = FeedForwardNN(intensity_w_layers_desc, name)
		self.add_nn = FeedForwardNN(add_layers_desc, name)
		self.erase_nn = FeedForwardNN(erase_layers_desc, name)
		self.allocation_nn = FeedForwardNN(allocation_layers_desc, name)
		self.write_gate_nn = FeedForwardNN(write_gate_layers_desc, name)

		self.free_gate_nn = FeedForwardNN(free_gate_layers_desc, name)

		self.key_r_nn = FeedForwardNN(key_r_layers_desc, name)
		self.intensity_r_nn = FeedForwardNN(intensity_r_layers_desc, name)
		self.read_mode_nn = FeedForwardNN(read_mode_layers_desc, name)
		



	def run_generation(self, inputs):

		key = self.key_w_nn.run_feed_forward_nn(inputs)
		intensity = self.intensity_w_nn.run_feed_forward_nn(inputs)
		add = self.add_nn.run_feed_forward_nn(inputs)
		erase = self.erase_nn.run_feed_forward_nn(inputs)
		allocation = self.allocation_nn.run_feed_forward_nn(inputs)
		write_gate = self.write_gate_nn.run_feed_forward_nn(inputs)

		free_gates = self.free_gate_nn.run_feed_forward_nn(inputs)

		keys = self.key_r_nn.run_feed_forward_nn(inputs)
		intensities = self.intensity_r_nn.run_feed_forward_nn(inputs)
		read_mode = self.read_mode_nn.run_feed_forward_nn(inputs)

		return key, intensity, add, erase, allocation, write_gate, free_gates, keys, intensities, read_mode

