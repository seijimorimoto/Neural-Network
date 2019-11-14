#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<NeuronLayer> &layers)
{
	this->layers = layers;
}

void NeuralNetwork::backPropagation()
{
	for (unsigned int i = this->layers.size() - 1; i >= 0; i--)
	{
		
	}
}


void NeuralNetwork::feedForward()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *prevLayerNeurons = this->layers[i - 1].neurons;
		this->layers[i].computeInputs(prevLayerNeurons);
		this->layers[i].computeActivationValues();
	}
}


void NeuralNetwork::initializeWeights()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		const unsigned int n = this->layers[i - 1].size();
		this->layers[i].initializeWeights(n);
	}
}


NeuralNetwork::~NeuralNetwork()
{
}
