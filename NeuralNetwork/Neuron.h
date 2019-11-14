#pragma once
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(int minWeight, int maxWeight);
	~Neuron();
	void computeInput(vector<Neuron> *previousLayerNeurons);
	void computeActivationValue(double lambda);
	void computeLocalGradient(vector<double> *weights, vector<double> localGradients);
	void initializeWeights(unsigned int n, int minWeight, int maxWeight);
	void setExpectedOutput(double expectedOutput);
	void setInputValue(double inputValue);
	void setWeightsSize(unsigned int n);
	double activationValue;
	double expectedOutput;
	double inputValue;
	double localGradient;
	vector<double> *weights;
};

