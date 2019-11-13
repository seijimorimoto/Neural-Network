#include "NeuronLayer.h"



NeuronLayer::NeuronLayer(int n, NeuronLayer *previousLayer, double lambda, vector<double> *inputs, int minWeight, int maxWeight)
{
	if (previousLayer)
	{
		this->neurons = new vector<Neuron>(n, Neuron(previousLayer->size(), minWeight, maxWeight));
	}
	else
	{
		this->neurons = new vector<Neuron>(n, Neuron(0, minWeight, maxWeight));
	}

	if (inputs)
	{
		for (unsigned int i = 0; i < inputs->size(); i++)
		{
			(*this->neurons)[i].setInputValue((*inputs)[i]);
		}
	}
	
	this->previousLayer = previousLayer;
	this->lambda = lambda;
	this->minWeight = minWeight;
	this->maxWeight = maxWeight;
}


NeuronLayer::~NeuronLayer()
{
}

int NeuronLayer::size()
{
	return this->neurons->size();
}

void NeuronLayer::computeInputs()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeInput(this->previousLayer->neurons);
	}
}

void NeuronLayer::computeActivationValues()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeActivationValue(lambda);
	}
}

