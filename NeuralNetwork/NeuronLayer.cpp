#include <iostream>
#include "NeuronLayer.h"

// Constructs a neuron layer (with sigmoid activation function).
// Params:
// - n: Number of regular neurons to be put in the layer.
// - biasN: Number of bias neurons to be put in the layer.
// - lambda: The lambda parameter for the sigmoid activation function.
// - minWeight: Minimum value for the weights between the neurons in this layer and the previous.
// - maxWeight: Maximum value for the weights between the neurons in this layer and the previous.
NeuronLayer::NeuronLayer(int n, int biasN, double lambda, int minWeight, int maxWeight)
{
	this->neurons = new vector<Neuron>(n + biasN, Neuron());
	this->biasN = biasN;
	this->lambda = lambda;
	this->minWeight = minWeight;
	this->maxWeight = maxWeight;
}

// Destructor of a neuron layer.
NeuronLayer::~NeuronLayer()
{
	/*delete this->neurons;*/
}

// Computes the activation values of all the non-bias neurons in the layer.
void NeuronLayer::computeActivationValues()
{
	for (unsigned int i = 0; i < this->neurons->size() - this->biasN; i++)
	{
		(*this->neurons)[i].computeActivationValue(lambda);
	}
}

// Computes the errors of all neurons in the layer. This should only be called when this layer is
// the output layer in a neural network.
void NeuronLayer::computeErrors()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeError();
	}
}

// Computes the input values of all neurons in this layer. 
// Params:
// - prevLayerNeurons: The neurons of the previous layer.
void NeuronLayer::computeInputs(vector<Neuron> *prevLayerNeurons)
{
	// Compute the input values of all non-bias neurons.
	for (unsigned int i = 0; i < this->neurons->size() - this->biasN; i++)
	{
		(*this->neurons)[i].computeInput(prevLayerNeurons);
	}

	// Set the input and activation values of all bias-neurons to be equal to 1.
	for (unsigned int i = this->neurons->size() - this->biasN; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].setInputValue(1);
	}
}

// Computes the local gradients of all neurons in this layer. This should be called when this layer
// is the output layer of a neural network.
void NeuronLayer::computeLocalGradients()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeLocalGradient(this->lambda);
	}
}

// Computes the local gradients of all neurons in this layer. This should be called when this layer
// is a hidden layer of a neural network.
// Params:
// - nextLayerNeurons: The neurons of the next layer in the neural network.
// - nextLayerBiasN: The number of neurons in the next layer that are biases.
void NeuronLayer::computeLocalGradients(vector<Neuron>* nextLayerNeurons, int nextLayerBiasN)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].computeLocalGradient(this->lambda, nextLayerNeurons, nextLayerBiasN, i);
	}
}

// Gets the error of the neuron at a given index in this layer.
// Params:
// - neuronIndex: The index of the neuron from which to return the error.
// Returns: The error of the neuron.
double NeuronLayer::getNeuronError(unsigned int neuronIndex)
{
	return (*this->neurons)[neuronIndex].error;
}

// Returns a vector with the activation values of all the neurons in this layer.
vector<double> NeuronLayer::getActivationValues()
{
	vector<double> activationValues;
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		activationValues.push_back((*this->neurons)[i].activationValue);
	}
	return activationValues;
}

// Returns a vector with the weights between each neuron in this layer and the previous one.
// The weights associated with a neuron in this layer are represented as a vector.
vector<vector<double>> NeuronLayer::getWeights()
{
	vector<vector<double>> neuronsWeights;
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		neuronsWeights.push_back((*this->neurons)[i].getWeights());
	}
	return neuronsWeights;
}

// Randomly initializes the weights between the neurons in this layer and the previous one.
// Params:
// - numberOfPrevLayerNeurons: The number of neurons that exist in the previous neuron layer.
void NeuronLayer::initializeWeights(unsigned int numberOfPrevLayerNeurons)
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].setWeightsSize(numberOfPrevLayerNeurons);
		(*this->neurons)[i].initializeWeights(numberOfPrevLayerNeurons, minWeight, maxWeight);
	}
}

// Prints the activation values of all the neurons in this layer.
void NeuronLayer::printActivationValues()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		cout << (*neurons)[i].activationValue << endl;
	}
}

// Prints the local gradients of all the neurons in this layer.
void NeuronLayer::printLocalGradients()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		cout << (*neurons)[i].localGradient << endl;
	}
}

// Prints all the weights between the neurons in this layer and the previous one.
void NeuronLayer::printWeights()
{
	for (unsigned int i = 0; i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].printWeights(i);
	}
}

// Sets the input values for the neurons in this layer. This should be called only when this layer
// is the input layer in a neural network.
// Params:
// - inputValues: Vector with the values to be set as inputs for the neurons in this layer.
void NeuronLayer::setInputValues(vector<double> &inputValues)
{
	// Set the input values to the non-bias neurons.
	for (unsigned int i = 0; i < inputValues.size(); i++)
	{
		(*this->neurons)[i].setInputValue(inputValues[i]);
	}

	// If there are more neurons than input values to be set, then give the remaining neurons an
	// input value of 1 (since they are biases).
	for (unsigned int i = inputValues.size(); i < this->neurons->size(); i++)
	{
		(*this->neurons)[i].setInputValue(1);
	}
}

// Sets the expected output values for the neurons in this layer. This should be called only when
// this layer is the output layer in a neural network.
// Params:
// - outputValues: Vector with the expected output values to be set for the neurons in this layer.
void NeuronLayer::setOutputValues(vector<double> &outputValues)
{
	for (unsigned int i = 0; i < outputValues.size(); i++)
	{
		(*this->neurons)[i].setExpectedOutput(outputValues[i]);
	}
}

// Sets the weights between a given neuron in this layer and the neurons in the previous layer.
// Params:
// - weights: The weights having as origin the neurons of the previous layer.
// - neuronIndex: The index of the neuron in this layer that is the destination for the weights
// passed as parameter.
void NeuronLayer::setWeight(vector<double>* weights, int neuronIndex)
{
	(*this->neurons)[neuronIndex].setWeights(weights);
}

// Updates the weights between each neuron in this layer and the previous one.
// Params:
// - prevLayerNeurons: The neurons in the previous layer.
// - learningRate: Value between 0 and 1 that determines how big will be a weight update operation.
// - momentum: Value between 0 and 1 that determines how much previous updates will contribute to
// the current weight update.
void NeuronLayer::updateWeights(vector<Neuron>* prevLayerNeurons, double learningRate, double momentum)
{
	// Just update the weights associated to the non-bias neurons, since bias neurons aren't linked
	// to neurons in the previous layer.
	for (unsigned int i = 0; i < this->neurons->size() - this->biasN; i++)
	{
		(*this->neurons)[i].updateWeights(prevLayerNeurons, learningRate, momentum);
	}
}

// Returns the number of neurons (including bias neurons) in this layer.
int NeuronLayer::size()
{
	return this->neurons->size();
}