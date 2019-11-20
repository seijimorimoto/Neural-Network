#pragma once
#include <vector>
#include "NeuronLayer.h"

using namespace std;

// TODO: Maybe include automatic finding of min and max for each input and output features for normalization.
class NeuralNetwork
{
public:
	NeuralNetwork(vector<NeuronLayer> &layers, double learningRate, double momentum);
	~NeuralNetwork();
	void accumulateStepErrors(vector<double> &accumErrors);
	void backPropagation();
	void exportModel(string filePath);
	void feedForward();
	void initializeWeights();
	double getEpochError(vector<double> &stepErrors, unsigned int trainSize);
	vector<double> getInputsFromDataRecord(vector<double> &dataRecord);
	vector<double> getOutputsFromDataRecord(vector<double> &dataRecord);
	void normalizeDataSet();
	void printActivationValues(); // used for testing purposes.
	void printDataSet(); // used for testing purposes.
	void printLocalGradients(); // used for testing purposes.
	void printWeights(); // used for testing purposes.
	void setCsvDataFile(string csvFilePath, unsigned int inputColumns, unsigned int outputColumns, unsigned int startRow, double trainPerc, double validationPerc);
	void setNormalizationValues(double inputMin, double inputMax, double outputMin, double outputMax);
	void setValuesToInputLayer(vector<double> &inputValues);
	void setValuesToOutputLayer(vector<double> &outputValues);
	void shuffleDataSet();
	void train(unsigned int epochs);
	vector<vector<double>> dataSet;
	unsigned int inputFeatures;
	vector<NeuronLayer> layers;
	double learningRate;
	double minInput;
	double minOutput;
	double maxInput;
	double maxOutput;
	double momentum;
	unsigned int outputFeatures;
	double trainPercentage;
	double validationPercentage;
};

