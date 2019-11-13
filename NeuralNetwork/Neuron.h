#pragma once
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(int prevLayerSize, int minWeight, int maxWeight);
	~Neuron();
	void computeInput(vector<Neuron> *previousLayerNeurons);
	void computeActivationValue(double lambda);
	void initializeWeights(int minWeight, int maxWeight);
	void setInputValue(double inputValue);
	double inputValue;
	double activationValue;
	vector<double> *weights;
};

