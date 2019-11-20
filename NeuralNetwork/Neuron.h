#pragma once
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron();
	~Neuron();
	void computeActivationValue(double lambda);
	void computeError();
	void computeInput(vector<Neuron> *previousLayerNeurons);
	void computeLocalGradient(double lambda);
	void computeLocalGradient(double lambda, vector<Neuron> *nextLayerNeurons, unsigned int neuronIndex);
	void initializeWeights(unsigned int n, int minWeight, int maxWeight);
	void printWeights(unsigned int neuronIndex); // used for testing purposes.
	void setExpectedOutput(double expectedOutput);
	void setInputValue(double inputValue);
	void setWeights(vector<double> *weights); // used for testing purposes.
	void setWeightsSize(unsigned int n);
	void updateWeights(vector<Neuron> *prevLayerNeurons, double learningRate, double momentum);
	double getWeight(unsigned int prevLayerNeuronIndex);
	double activationValue;
	double error;
	double expectedOutput;
	double inputValue;
	double localGradient;
	vector<double> *weights;
	vector<double> *weightsDelta;
};

