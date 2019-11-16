#include <ctime>
#include <iostream>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<NeuronLayer> &layers, double learningRate)
{
	this->layers = layers;
	this->learningRate = learningRate;
}


NeuralNetwork::~NeuralNetwork()
{
}


void NeuralNetwork::backPropagation()
{
	unsigned int i = this->layers.size() - 1;
	this->layers[i].computeErrors();
	this->layers[i].computeLocalGradients();
	for (i = i - 1; i > 0; i--)
	{
		vector<Neuron> *neurons = this->layers[i + 1].neurons;
		this->layers[i].computeLocalGradients(neurons);
	}
	for (i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *neurons = this->layers[i - 1].neurons;
		this->layers[i].updateWeights(neurons, this->learningRate);
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
	srand(time(nullptr));
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		const unsigned int n = this->layers[i - 1].size();
		this->layers[i].initializeWeights(n);
	}
}

void NeuralNetwork::printActivationValues()
{
	for (unsigned int i = 0; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printActivationValues();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::printLocalGradients()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printLocalGradients();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::printWeights()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printWeights();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::train(int epochs)
{
	for (unsigned int i = 0; i < epochs; i++)
	{
		feedForward();
		backPropagation();
	}
}
