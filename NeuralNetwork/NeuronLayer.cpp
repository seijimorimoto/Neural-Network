#include <iostream>
#include "NeuronLayer.h"


NeuronLayer::NeuronLayer(int n, double lambda, int minWeight, int maxWeight)
{
	this->neurons = new vector<Neuron>(n, Neuron());
	this->lambda = lambda;
	this->minWeight = minWeight;
	this->maxWeight = maxWeight;
}

NeuronLayer::NeuronLayer(int n, double lambda, vector<double> *inputs, vector<double> *outputs, int minWeight, int maxWeight)
{
	this->neurons = new vector<Neuron>(n, Neuron());
	this->lambda = lambda;
	this->minWeight = minWeight;
	this->maxWeight = maxWeight;

	if (inputs)
	{
		setInputValues(*inputs);
	}

	if (outputs)
	{
		setOutputValues(*outputs);
	}
}


NeuronLayer::~NeuronLayer()
{
	delete this->neurons;
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

double NeuronLayer::getNeuronError(unsigned int neuronIndex)
{
	return (*this->neurons)[neuronIndex].error;
}

vector<double> NeuronLayer::getActivationValues()
{
	vector<double> activationValues;
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		activationValues.push_back((*this->neurons)[i].activationValue);
	}
	return activationValues;
}

vector<vector<double>> NeuronLayer::getWeights()
{
	vector<vector<double>> neuronsWeights;
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		neuronsWeights.push_back((*this->neurons)[i].getWeights());
	}
	return neuronsWeights;
}


void NeuronLayer::initializeWeights(unsigned int numberOfPrevLayerNeurons)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].setWeightsSize(numberOfPrevLayerNeurons);
		(*this->neurons)[i].initializeWeights(numberOfPrevLayerNeurons, minWeight, maxWeight);
	}
}

void NeuronLayer::printActivationValues()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		cout << (*neurons)[i].activationValue << endl;
	}
}

void NeuronLayer::printLocalGradients()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		cout << (*neurons)[i].localGradient << endl;
	}
}

void NeuronLayer::printWeights()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].printWeights(i);
	}
}

void NeuronLayer::setInputValues(vector<double> &inputValues)
{
	for (unsigned int i = 0; i < inputValues.size(); i++)
	{
		(*this->neurons)[i].setInputValue(inputValues[i]);
	}
}

void NeuronLayer::setOutputValues(vector<double> &outputValues)
{
	for (unsigned int i = 0; i < outputValues.size(); i++)
	{
		(*this->neurons)[i].setExpectedOutput(outputValues[i]);
	}
}


void NeuronLayer::setWeight(vector<double>* weights, int neuronIndex)
{
	(*this->neurons)[neuronIndex].setWeights(weights);
}


void NeuronLayer::updateWeights(vector<Neuron>* prevLayerNeurons, double learningRate, double momentum)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].updateWeights(prevLayerNeurons, learningRate, momentum);
	}
}


int NeuronLayer::size()
{
	return this->neurons->size();
}