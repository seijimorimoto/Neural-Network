#include <cmath>
#include <cstdlib>
#include <ctime>
#include "Neuron.h"

Neuron::Neuron(int minWeight, int maxWeight)
{
	/*this->weights = new vector<double>(prevLayerSize);
	initializeWeights(minWeight, maxWeight);*/
	//}
	this->inputValue = 0;
	this->activationValue = 0;
}


Neuron::~Neuron()
{
}

void Neuron::computeInput(vector<Neuron> *previousLayerNeurons)
{
	for (unsigned int i = 0; i < previousLayerNeurons->size(); i++)
	{
		this->inputValue += (*this->weights)[i] * (*previousLayerNeurons)[i].activationValue;
	}
}

void Neuron::computeActivationValue(double lambda)
{
	this->activationValue = 1 / (1 + exp(-lambda * this->inputValue));
}

void Neuron::initializeWeights(unsigned int n, int minWeight, int maxWeight)
{
	srand(time(nullptr));
	const int weightRange = maxWeight - minWeight;
	for (unsigned int i = 0; i < this->weights->size(); i++)
	{
		(*this->weights)[i] = minWeight + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / weightRange));
	}
	
}

void Neuron::setExpectedOutput(double expectedOutput)
{
	this->expectedOutput = expectedOutput;
}

void Neuron::setInputValue(double inputValue)
{
	this->inputValue = inputValue;
	this->activationValue = inputValue;
}


void Neuron::setWeightsSize(unsigned int n)
{
	this->weights = new vector<double>(n);
}

