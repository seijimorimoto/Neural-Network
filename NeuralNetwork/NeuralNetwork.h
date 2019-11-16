#pragma once
#include <vector>
#include "NeuronLayer.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(vector<NeuronLayer> &layers, double learningRate, double momentum);
	~NeuralNetwork();
	void backPropagation();
	void feedForward();
	void initializeWeights();
	void printActivationValues(); // used for testing purposes.
	void printDataSet(); // used for testing purposes.
	void printLocalGradients(); // used for testing purposes.
	void printWeights(); // used for testing purposes.
	void setCsvDataFile(string csvFilePath, unsigned int inputColumns, unsigned int outputColumns, unsigned int startRow);
	void train(int epochs);
	vector<vector<double>> inputData;
	vector<NeuronLayer> layers;
	double learningRate;
	double momentum;
	vector<vector<double>> outputData;
};

