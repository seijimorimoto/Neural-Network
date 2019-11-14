#pragma once
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(int minWeight, int maxWeight);
	~Neuron();
	void computeActivationValue(double lambda);
	void computeError();
	void computeInput(vector<Neuron> *previousLayerNeurons);
	void computeLocalGradient(double lambda);
	void computeLocalGradient(double lambda, vector<Neuron> *nextLayerNeurons, unsigned int neuronIndex);
	void initializeWeights(unsigned int n, int minWeight, int maxWeight);
	void setExpectedOutput(double expectedOutput);
	void setInputValue(double inputValue);
	void setWeightsSize(unsigned int n);
	void updateWeights(vector<Neuron> *prevLayerNeurons, double learningRate);
	double getWeight(unsigned int prevLayerNeuronIndex);
	double activationValue;
	double error;
	double expectedOutput;
	double inputValue;
	double localGradient;
	vector<double> *weights;
};

