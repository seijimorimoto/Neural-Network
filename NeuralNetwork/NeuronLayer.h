#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuronLayer
{
public:
	NeuronLayer(int n, NeuronLayer *previousLayer, double lambda, vector<double> *inputs=nullptr, int minWeight=0, int maxWeight=1);
	~NeuronLayer();
	void computeInputs();
	void computeActivationValues();
	int size();
	double lambda;
	int minWeight;
	int maxWeight;
	NeuronLayer *previousLayer;
	vector<Neuron> *neurons;
};

