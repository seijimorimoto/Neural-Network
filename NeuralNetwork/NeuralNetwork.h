#pragma once
#include <vector>
#include "NeuronLayer.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(vector<NeuronLayer> &layers, double learningRate);
	~NeuralNetwork();
	void backPropagation();
	void feedForward();
	void initializeWeights();
	void printActivationValues(); // used for testing purposes.
	void printLocalGradients(); // used for testing purposes.
	void printWeights(); // used for testing purposes.
	void train(int epochs);
	vector<NeuronLayer> layers;
	double learningRate;
};

