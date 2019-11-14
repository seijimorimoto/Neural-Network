#pragma once
#include <vector>
#include "NeuronLayer.h"
class NeuralNetwork
{
public:
	NeuralNetwork(vector<NeuronLayer> &layers);
	void backPropagation();
	void feedForward();
	void initializeWeights();
	vector<NeuronLayer> layers;
	~NeuralNetwork();
};

