#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuronLayer
{
public:
	NeuronLayer(int n, int biasN, double lambda, int minWeight = 0, int maxWeight = 1);
	~NeuronLayer();
	void computeActivationValues();
	void computeErrors();
	void computeInputs(vector<Neuron> *prevLayerNeurons);
	void computeLocalGradients();
	void computeLocalGradients(vector<Neuron> *nextLayerNeurons, int nextLayerBiasN);
	double getNeuronError(unsigned int neuronIndex);
	vector<double> getActivationValues();
	vector<vector<double>> getWeights();
	void initializeWeights(unsigned int numberOfPrevLayerNeurons);
	void printActivationValues(); // used for testing purposes.
	void printLocalGradients(); // used for testing purposes.
	void printWeights(); // used for testing purposes.
	void setInputValues(vector<double> &inputValues);
	void setOutputValues(vector<double> &outputValues);
	void setWeight(vector<double> *weights, int neuronIndex);
	void updateWeights(vector<Neuron> *prevLayerNeurons, double learningRate, double momentum);
	int size();

	int biasN;
	double lambda;
	int minWeight;
	int maxWeight;
	vector<Neuron> *neurons;
};

