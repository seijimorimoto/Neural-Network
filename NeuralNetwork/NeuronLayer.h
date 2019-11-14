#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuronLayer
{
public:
	NeuronLayer(int n, double lambda, vector<double> *inputs=nullptr, vector<double> *outputs=nullptr, int minWeight=0, int maxWeight=1);
	~NeuronLayer();
	void computeInputs(vector<Neuron> *prevLayerNeurons);
	void computeActivationValues();
	void initializeWeights(unsigned int numberOfPrevLayerNeurons);
	int size();
	double lambda;
	int minWeight;
	int maxWeight;
	vector<Neuron> *neurons;
};

