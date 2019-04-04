from random import *

class IAFNeuron:
	def __init__(self, r=1):
		self.x=0
		self.r=r
	def __call__(self, x, dt):
		self.x+=x-self.x*self.r*dt
		self.x=0 if self.x < 0 else self.x
		return self.x

class IAFNetwork:
	def __init__(self, inputs, inc_rate, dec_rate, deccel_rate, max_weight, min_weight, min_value, max_value):
		self.inc_rate=inc_rate
		self.dec_rate=dec_rate
		self.max_weight=max_weight
		self.min_weight=min_weight
		self.deccel_rate=deccel_rate
		self.min_value=min_value
		self.max_value=max_value
		self.neurons=[IAFNeuron(self.deccel_rate) for i in range(inputs)]
		self.sources=[[] for i in range(inputs)]
		self.weights=[[] for i in range(inputs)]
		self.values=[0 for i in range(inputs)]
		self.current=[0 for i in range(inputs)]
		self.outputs=[0 for i in range(inputs)]
		self.inputs=[0 for i in range(inputs)]
		self.states=[0 for i in range(inputs)]
		self.thresholds=[0 for i in range(inputs)]
		self.previous=[0 for i in range(inputs)]
		self.input_neurons=[i for i in range(inputs)]
		self.inter_neurons=[]
		self.output_neurons=[]
		self.active_neurons=[]

	def create_neuron(self, threshold, sources=[], weights=[], isoutput=False):
		self.neurons.append(IAFNeuron(self.deccel_rate))
		self.thresholds.append(threshold)
		self.sources.append([])
		self.weights.append([])
		self.values.append(0)
		self.outputs.append(0)
		self.current.append(0)
		self.previous.append(0)
		self.states.append(0)
		self.inputs.append(0)
		index=len(self.neurons)-1
		self.inter_neurons.append(index)
		if isoutput:self.output_neurons.append(index)
		self.append_links(index, sources, weights)
		return index

	def append_links(self, index, sources, weights):
		if index in sources:
			i=sources.index(index)
			del sources[i]
			del weights[i]
		self.sources[index]+=sources
		self.weights[index]+=weights

	def update_neuron(self, index, input, dt):
		threshold=self.thresholds[index]
		neuron=self.neurons[index]
		state=self.states[index]
		x=input if state in [0,1] else 0
		value=neuron(x, dt)
		if state==0 and value >= threshold:state=1
		elif state==1 and value >= self.max_value:state=2
		elif state==2 and value <= threshold:state=3
		elif state==3 and value <= self.min_value:state=0
		y=1 if state in [1,2] else 0
		self.inputs[index]=x
		self.current[index]=y
		self.values[index]=value
		self.neurons[index]=neuron
		self.states[index]=state

	def compute_neuron(self, index, dt):
		threshold=self.thresholds[index]
		neuron=self.neurons[index]
		sources=self.sources[index]
		weights=self.weights[index]
		state=self.states[index]
		x=0
		for i in range(len(sources)):
			source=sources[i]
			w=weights[i]
			y=self.outputs[source]
			x+=y*w
		self.update_neuron(index, x, dt)

	def train_neuron(self, index, dt):
		sources=self.sources[index]
		weights=self.weights[index]
		output=self.current[index]
		for i in range(len(sources)):
			source=sources[i]
			w=weights[i]
			y=self.outputs[source]
			delta=y*self.inc_rate
			if w!=0:delta-=w/abs(w)*self.dec_rate
			w+=output*delta*dt
			if w > self.max_weight:w=self.max_weight
			if w < self.min_weight:w=self.min_weight
			self.weights[index][i]=w
		v=self.values[index]
		t=self.thresholds[index]
		delta=(v-t)*self.inc_rate
		t+=delta*dt
		if t < 0:t=0
		self.thresholds[index]=t

	def update_inputs(self, inputs, dt):
		for i in range(len(self.input_neurons)):
			input=inputs[i]
			index=self.input_neurons[i]
			self.update_neuron(index, input, dt)

	def compute_neurons(self, dt):
		for index in self.inter_neurons:
			self.compute_neuron(index, dt)

	def train_neurons(self, dt):
		for index in self.inter_neurons:
			self.train_neuron(index, dt)

	def retrieve_outputs(self):
		outputs=[]
		for index in self.output_neurons:
			output=self.outputs[index]
			outputs.append(output)
		return outputs

	def compute(self, inputs, dt):
		self.previous=self.outputs
		self.update_inputs(inputs, dt)
		self.compute_neurons(dt)
		self.train_neurons(dt)
		self.outputs=self.current
		outputs=self.retrieve_outputs()
		return outputs

def create_network(inc_rate, dec_rate, deccel_rate, max_weight, min_weight, min_value, max_value, layer_sizes, layer_connections):
	network=IAFNetwork(
		inputs=layer_sizes[0], 
		inc_rate=inc_rate, 
		dec_rate=dec_rate,
		deccel_rate=deccel_rate,
		max_weight=max_weight, 
		min_weight=min_weight,
		min_value=min_value,
		max_value=max_value)

	layer_indices=[network.input_neurons]
	for i in range(1, len(layer_sizes)):
		isoutput=len(layer_sizes)-1==i
		indices=[]
		for j in range(layer_sizes[i]):
			index=network.create_neuron(0, isoutput=isoutput)
			indices.append(index)
		layer_indices.append(indices)
	for i in range(1, len(layer_connections)):
		for j in range(len(layer_connections[i])):
			layer_index=layer_connections[i][j]
			inputs=layer_indices[layer_index]
			for index in layer_indices[i]:
				weights=[random() for k in range(len(inputs))]
				network.append_links(index, inputs, weights)
	return network