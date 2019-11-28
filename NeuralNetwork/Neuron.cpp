#include <cmath>
#include <iostream>
#include "Neuron.h"

// Constructs a neuron.
Neuron::Neuron()
{
	this->inputValue = 0;
	this->activationValue = 0;
}

// Destructor of a neuron.
Neuron::~Neuron()
{
	/*delete this->weights;
	delete this->weightsDelta;*/
}

// Computes the activation value of the neuron (using a sigmoid activation function).
// Params:
// - lambda: The lambda parameter for the sigmoid activation function.
void Neuron::computeActivationValue(double lambda)
{
	this->activationValue = 1 / (1 + exp(-lambda * this->inputValue));
}

// Computes the error of the neuron. This should only be called if an expectedOutput of the neuron
// has been previously set (i.e. this is a neuron in the output layer of a neural network).
void Neuron::computeError()
{
	this->error = this->expectedOutput - this->activationValue;
}

// Computes the input value of the neuron from the activation values of the neurons in the previous
// layer of the neural network and the weights associated with them.
// Params:
// - previousLayerNeurons: The neurons in the previous layer of the neural network.
void Neuron::computeInput(vector<Neuron> *previousLayerNeurons)
{
	this->inputValue = 0;
	for (unsigned int i = 0; i < previousLayerNeurons->size(); i++)
	{
		this->inputValue += (*this->weights)[i] * (*previousLayerNeurons)[i].activationValue;
	}
}

// Computes the local gradient of the neuron (if it is an output neuron).
// Params:
// - lambda: The lambda parameter used in the sigmoid activation function.
void Neuron::computeLocalGradient(double lambda)
{
	this->localGradient = lambda * this->activationValue * (1 - this->activationValue) * this->error;
}

// Computes the local gradient of the neuron (if it is a hidden neuron).
// Params:
// - lambda: The lambda parameter used in the sigmoid activation function.
// - nextLayerNeurons: The neurons in the next layer of the neural network.
// - nextLayerBiasN: The number of neurons in the next layer of the neural network that are biases.
// - neuronIndex: The index of this neuron within its neuron layer.
void Neuron::computeLocalGradient(double lambda, vector<Neuron> *nextLayerNeurons, int nextLayerBiasN, unsigned int neuronIndex)
{
	double gradientWeightSum = 0;
	for (unsigned int i = 0; i < nextLayerNeurons->size() - nextLayerBiasN; i++)
	{
		gradientWeightSum += (*nextLayerNeurons)[i].localGradient * (*nextLayerNeurons)[i].getWeight(neuronIndex);
	}
	this->localGradient = lambda * this->activationValue * (1 - this->activationValue) * gradientWeightSum;
}

// Gets the weight between this neuron and a neuron from the previous layer of the neural network.
// Params:
// - prevLayerNeuronIndex: Index of the neuron in the previous layer.
// Returns: The weight between this neuron and the neuron from the previous layer.
double Neuron::getWeight(unsigned int prevLayerNeuronIndex)
{
	return (*this->weights)[prevLayerNeuronIndex];
}

// Returns the weights between this neuron and the neurons in the previous layer of neural network.
vector<double> Neuron::getWeights()
{
	return *this->weights;
}

// Initializes the weights between this neuron and the neurons in the previous layer of the neural
// network.
// Params:
// - minWeight: The minimum value that a weight can have.
// - maxWeight: The maximum value that a weight can have.
void Neuron::initializeWeights(int minWeight, int maxWeight)
{
	const int weightRange = maxWeight - minWeight;
	for (unsigned int i = 0; i < this->weights->size(); i++)
	{
		(*this->weights)[i] = minWeight + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / weightRange));
	}
	
}

// Prints the weights between this neuron and the neurons in the previous layer of the neural
// network.
// Params:
// - neuronIndex: The index of this neuron within its neuron layer.
void Neuron::printWeights(unsigned int neuronIndex)
{
	for (unsigned int i = 0; i < this->weights->size(); i++)
	{
		cout << "  W(" << neuronIndex << "," << i << "): " << (*this->weights)[i] << endl;
	}
}

// Sets the expected output of this neuron. This should be used if this neuron is an output neuron.
// Params:
// - expectedOutput: The expected output to be set in the neuron.
void Neuron::setExpectedOutput(double expectedOutput)
{
	this->expectedOutput = expectedOutput;
}

// Sets the input value of this neuron. This should be used if this neuron is from the input layer.
// Params:
// - inputValue: The input value to be set in the neuron. 
void Neuron::setInputValue(double inputValue)
{
	this->inputValue = inputValue;
	this->activationValue = inputValue;
}

// Sets the weights between this neuron and the neurons in the previous layer of neural network.
// Params:
// - weights: Vector with the values of the weights between this neuron and each neuron in the
// previous layer of the neural network.
void Neuron::setWeights(vector<double>* weights)
{
	this->weights = weights;
	this->weightsDelta = new vector<double>(weights->size(), 0);
}

// Sets the number of weights associated with this neuron and the previous layer.
// Params:
// - n: The number of weights associated with this neuron and the previous layer (i.e. the number
// of neurons in the previous layer).
void Neuron::setWeightsSize(unsigned int n)
{
	this->weights = new vector<double>(n);
	this->weightsDelta = new vector<double>(n, 0);
}

// Updates the weights between this neuron and the ones in previous layer of the neural network.
// Params:
// - prevLayerNeurons: The neurons from the previous layer in the neural network.
// - learningRate: Value between 0 and 1 that determines how big will be a weight update operation.
// - momentum: Value between 0 and 1 that determines how much previous updates will contribute to
// the current weight update.
void Neuron::updateWeights(vector<Neuron>* prevLayerNeurons, double learningRate, double momentum)
{
	for (unsigned int i = 0; i < this->weights->size(); i++)
	{
		double deltaWeight = learningRate * this->localGradient * (*prevLayerNeurons)[i].activationValue + momentum * (*this->weightsDelta)[i];
		(*this->weights)[i] += deltaWeight;
		(*this->weightsDelta)[i] = deltaWeight;
	}
}