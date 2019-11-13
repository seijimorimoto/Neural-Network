#include <cmath>
#include <cstdlib>
#include <ctime>
#include "Neuron.h"

Neuron::Neuron(int prevLayerSize, int minWeight, int maxWeight)
{
	if (prevLayerSize > 0)
	{
		this->weights = new vector<double>(prevLayerSize);
		initializeWeights(minWeight, maxWeight);
	}
	else
	{
		this->weights = nullptr;
	}
	this->inputValue = 0;
	this->activationValue = 0;
}


Neuron::~Neuron()
{
}

void Neuron::computeInput(vector<Neuron> *previousLayerNeurons)
{
	if (this->weights)
	{
		for (unsigned int i = 0; i < previousLayerNeurons->size(); i++)
		{
			this->inputValue += (*this->weights)[i] * (*previousLayerNeurons)[i].activationValue;
		}
	}
}

void Neuron::computeActivationValue(double lambda)
{
	if (this->weights)
	{
		this->activationValue = 1 / (1 + exp(-lambda * this->inputValue));
	}
	else
	{
		this->activationValue = this->inputValue;
	}
}

void Neuron::initializeWeights(int minWeight, int maxWeight)
{
	srand(time(nullptr));
	const int weightRange = maxWeight - minWeight;
	for (unsigned int i = 0; i < this->weights->size(); i++)
	{
		(*this->weights)[i] = minWeight + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / weightRange));
	}
	
}

void Neuron::setInputValue(double inputValue)
{
	this->inputValue = inputValue;
}
