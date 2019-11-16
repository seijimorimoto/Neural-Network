#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuronLayer
{
public:
	NeuronLayer(int n, double lambda, vector<double> *inputs=nullptr, vector<double> *outputs=nullptr, int minWeight=0, int maxWeight=1);
	~NeuronLayer();
	void computeActivationValues();
	void computeErrors();
	void computeInputs(vector<Neuron> *prevLayerNeurons);
	void computeLocalGradients();
	void computeLocalGradients(vector<Neuron> *nextLayerNeurons);
	void initializeWeights(unsigned int numberOfPrevLayerNeurons);
	void printActivationValues(); // used for testing purposes.
	void printLocalGradients(); // used for testing purposes.
	void printWeights(); // used for testing purposes.
	void setWeight(vector<double> *weights, int neuronIndex); // used for testing purposes.
	void updateWeights(vector<Neuron> *prevLayerNeurons, double learningRate);
	int size();
	double lambda;
	int minWeight;
	int maxWeight;
	vector<Neuron> *neurons;
};

