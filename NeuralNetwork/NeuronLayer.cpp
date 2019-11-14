#include "NeuronLayer.h"


NeuronLayer::NeuronLayer(int n, double lambda, vector<double> *inputs, vector<double> *outputs, int minWeight, int maxWeight)
{
	this->neurons = new vector<Neuron>(n, Neuron(minWeight, maxWeight));

	if (inputs)
	{
		for (unsigned int i = 0; i < inputs->size(); i++)
		{
			(*this->neurons)[i].setInputValue((*inputs)[i]);
		}
	}

	if (outputs)
	{
		for (unsigned int i = 0; i < outputs->size(); i++)
		{
			(*this->neurons)[i].expectedOutput = (*outputs)[i];
		}
	}
	
	this->lambda = lambda;
	this->minWeight = minWeight;
	this->maxWeight = maxWeight;
}


NeuronLayer::~NeuronLayer()
{
}


void NeuronLayer::computeActivationValues()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeActivationValue(lambda);
	}
}


void NeuronLayer::computeErrors()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeError();
	}
}


void NeuronLayer::computeInputs(vector<Neuron> *prevLayerNeurons)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeInput(prevLayerNeurons);
	}
}


void NeuronLayer::computeLocalGradients()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeLocalGradient(this->lambda);
	}
}


void NeuronLayer::computeLocalGradients(vector<Neuron>* nextLayerNeurons)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeLocalGradient(this->lambda, nextLayerNeurons, i);
	}
}


void NeuronLayer::initializeWeights(unsigned int numberOfPrevLayerNeurons)
{
	for (unsigned int i = 0; i < numberOfPrevLayerNeurons; i++)
	{
		(*this->neurons)[i].setWeightsSize(numberOfPrevLayerNeurons);
		(*this->neurons)[i].initializeWeights(numberOfPrevLayerNeurons, minWeight, maxWeight);
	}
}

void NeuronLayer::updateWeights(vector<Neuron>* prevLayerNeurons, double learningRate)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].updateWeights(prevLayerNeurons, learningRate);
	}
}


int NeuronLayer::size()
{
	return this->neurons->size();
}