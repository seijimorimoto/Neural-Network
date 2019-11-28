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
	void computeLocalGradient(double lambda, vector<Neuron> *nextLayerNeurons, int nextLayerBiasN, unsigned int neuronIndex);
	double getWeight(unsigned int prevLayerNeuronIndex);
	vector<double> getWeights();
	void initializeWeights(int minWeight, int maxWeight);
	void printWeights(unsigned int neuronIndex); // used for testing purposes.
	void setExpectedOutput(double expectedOutput);
	void setInputValue(double inputValue);
	void setWeights(vector<double> *weights);
	void setWeightsSize(unsigned int n);
	void updateWeights(vector<Neuron> *prevLayerNeurons, double learningRate, double momentum);
	
	double activationValue;
	double error;
	double expectedOutput;
	double inputValue;
	double localGradient;
	vector<double> *weights;
	vector<double> *weightsDelta;
};

