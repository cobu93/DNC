class LayerDescription():

	def __init__(self, size = 0, name = 'controller_layer', has_bias = True, is_input = False, is_output = False, dtype = None, activation = None):
		self.size = size
		self.name = name
		self.is_input = is_input
		self.is_output = is_output
		self.activation = activation
		self.has_bias = has_bias


class InputLayerDescription(LayerDescription):

	def __init__(self, name = ''):
		LayerDescription.__init__(self, size=0, name=name, is_input = True, has_bias=False)



class OutputLayerDescription(LayerDescription):

	def __init__(self, size = 0, name = '', activation = None, has_bias=True):
		LayerDescription.__init__(self, size=size, name=name, is_output = True, activation=activation, has_bias=has_bias)


class HiddenLayerDescription(LayerDescription):

	def __init__(self, size = 0, name = '', has_bias = True, activation = None):
		LayerDescription.__init__(self, size=size, name=name, activation=activation, has_bias=has_bias)