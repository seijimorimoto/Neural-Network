#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<NeuronLayer> &layers, double learningRate, double momentum)
{
	this->layers = layers;
	this->learningRate = learningRate;
	this->momentum = momentum;
}


NeuralNetwork::~NeuralNetwork()
{
}


void NeuralNetwork::accumulateStepErrors(vector<double>& accumErrors)
{
	for (unsigned int i = 0; i < accumErrors.size(); i++)
	{
		auto neuronError = this->layers[this->layers.size() - 1].getNeuronError(i);
		accumErrors[i] += neuronError * neuronError;
	}
}


void NeuralNetwork::backPropagation()
{
	unsigned int i = this->layers.size() - 1;
	this->layers[i].computeErrors();
	this->layers[i].computeLocalGradients();
	for (i = i - 1; i > 0; i--)
	{
		vector<Neuron> *neurons = this->layers[i + 1].neurons;
		this->layers[i].computeLocalGradients(neurons);
	}
	for (i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *neurons = this->layers[i - 1].neurons;
		this->layers[i].updateWeights(neurons, this->learningRate, this->momentum);
	}
}

void NeuralNetwork::exportModel(string filePath)
{
	ofstream file;
	file.open(filePath);

	if (file.is_open())
	{
		file << "LAYERS\n";
		for (unsigned int i = 0; i < this->layers.size(); i++)
		{
			auto neurons = this->layers[i].size();
			auto lambda = this->layers[i].lambda;
			auto minWeight = this->layers[i].minWeight;
			auto maxWeight = this->layers[i].maxWeight;
			file << neurons << "," << lambda << "," << minWeight << "," << maxWeight << "\n";
		}
		file << "\n";
		
		file << "WEIGHTS\n";
		for (unsigned int i = 1; i < this->layers.size(); i++)
		{
			auto neuronsWeights = this->layers[i].getWeights();
			for (unsigned int j = 0; j < neuronsWeights.size(); j++)
			{
				file << i << "," << j;
				for (unsigned int k = 0; k < neuronsWeights[j].size(); k++)
				{
					file << "," << neuronsWeights[j][k];
				}
				file << "\n";
			}
		}
		file << "\n";

		file << "PARAMS\n";
		file << "LEARNING_RATE," << this->learningRate << "\n";
		file << "MIN_INPUT," << this->minInput << "\n";
		file << "MIN_OUTPUT," << this->minOutput << "\n";
		file << "MAX_INPUT," << this->maxInput << "\n";
		file << "MAX_OUTPUT," << this->maxOutput << "\n";
		file << "MOMENTUM," << this->momentum << "\n";
		file.close();
	}
}


void NeuralNetwork::feedForward()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *prevLayerNeurons = this->layers[i - 1].neurons;
		this->layers[i].computeInputs(prevLayerNeurons);
		this->layers[i].computeActivationValues();
	}
}

NeuralNetwork NeuralNetwork::importModel(string filePath)
{
	ifstream file;
	file.open(filePath);

	if (file.is_open())
	{
		double learningRate, minInput, minOutput, maxInput, maxOutput, momentum;
		vector<NeuronLayer> layers;
		string line;
		unsigned int phase = 0;
		
		while (getline(file, line))
		{
			if (line == "LAYERS")
			{
				phase = 0;
				continue;
			}
			else if (line == "WEIGHTS")
			{
				phase = 1;
				continue;
			}
			else if (line == "PARAMS")
			{
				phase = 2;
				continue;
			}
			else if (line == "")
				continue;

			stringstream ss(line);
			switch (phase)
			{
				case 0:
				{
					string neurons, lambda, minWeight, maxWeight;
					getline(ss, neurons, ',');
					getline(ss, lambda, ',');
					getline(ss, minWeight, ',');
					getline(ss, maxWeight, ',');
					// TODO: Change type of minWeight and maxWeight to double.
					layers.push_back(NeuronLayer(stoi(neurons), stod(lambda), stod(minWeight), stod(maxWeight)));
					break;
				}
				case 1:
				{
					string layer, neuron, weight;
					vector<double> *neuronWeights = new vector<double>();
					getline(ss, layer, ',');
					getline(ss, neuron, ',');
					while (getline(ss, weight, ','))
					{
						neuronWeights->push_back(stod(weight));
					}
					layers[stoi(layer)].setWeight(neuronWeights, stoi(neuron));
					break;
				}
				case 2:
				{
					string paramLabel, paramValue;
					getline(ss, paramLabel, ',');
					getline(ss, paramValue, ',');
					if (paramLabel == "LEARNING_RATE")
						learningRate = stod(paramValue);
					else if (paramLabel == "MIN_INPUT")
						minInput = stod(paramValue);
					else if (paramLabel == "MIN_OUTPUT")
						minOutput = stod(paramValue);
					else if (paramLabel == "MAX_INPUT")
						maxInput = stod(paramValue);
					else if (paramLabel == "MAX_OUTPUT")
						maxOutput = stod(paramValue);
					else if (paramLabel == "MOMENTUM")
						momentum = stod(paramValue);
					break;
				}
			}
		}

		file.close();
		NeuralNetwork network(layers, learningRate, momentum);
		network.setNormalizationValues(minInput, maxInput, minOutput, maxOutput);
		return network;
	}
}


void NeuralNetwork::initializeWeights()
{
	srand(static_cast<unsigned int>(time(nullptr)));
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		const unsigned int n = this->layers[i - 1].size();
		this->layers[i].initializeWeights(n);
	}
}

double NeuralNetwork::getEpochError(vector<double> &stepErrors, unsigned int trainSize)
{
	double epochError = 0;
	for (unsigned int i = 0; i < stepErrors.size(); i++)
	{
		epochError += sqrt(stepErrors[i] / trainSize);
	}
	epochError /= stepErrors.size();
	return epochError;
}

vector<double> NeuralNetwork::getInputsFromDataRecord(vector<double> &dataRecord)
{
	vector<double> inputs;
	for (unsigned int i = 0; i < this->inputFeatures; i++)
	{
		inputs.push_back(dataRecord[i]);
	}
	return inputs;
}

vector<double> NeuralNetwork::getOutputsFromDataRecord(vector<double> &dataRecord)
{
	vector<double> outputs;
	for (unsigned int i = this->inputFeatures; i < this->inputFeatures + this->outputFeatures; i++)
	{
		outputs.push_back(dataRecord[i]);
	}
	return outputs;
}

void NeuralNetwork::normalizeDataSet()
{
	for (unsigned int i = 0; i < this->dataSet.size(); i++)
	{
		for (unsigned int j = 0; j < this->inputFeatures; j++)
		{
			this->dataSet[i][j] = (this->dataSet[i][j] - this->minInput) / (this->maxInput - this->minInput);
		}

		for (unsigned int j = this->inputFeatures; j < this->inputFeatures + this->outputFeatures; j++)
		{
			this->dataSet[i][j] = (this->dataSet[i][j] - this->minOutput) / (this->maxOutput - this->minOutput);
		}
	}
}

void NeuralNetwork::printActivationValues()
{
	for (unsigned int i = 0; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printActivationValues();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::printDataSet()
{
	for (unsigned int i = 0; i < this->dataSet.size(); i++)
	{
		for (unsigned int j = 0; j < this->inputFeatures; j++)
		{
			cout << this->dataSet[i][j] << " ";
		}
		cout << " | ";
		for (unsigned int j = this->inputFeatures; j < this->inputFeatures + this->outputFeatures; j++)
		{
			cout << " " << this->dataSet[i][j];
		}
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::printLocalGradients()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printLocalGradients();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::printWeights()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printWeights();
		cout << endl;
	}
	cout << endl;
}

void NeuralNetwork::setCsvDataFile(string csvFilePath, unsigned int inputColumns, unsigned int outputColumns, unsigned int startRow, double trainPerc, double validationPerc)
{
	ifstream file;
	file.open(csvFilePath);
	this->inputFeatures = inputColumns;
	this->outputFeatures = outputColumns;
	this->trainPercentage = trainPerc;
	this->validationPercentage = validationPerc;

	if (file.is_open())
	{
		string line;
		for (unsigned int i = 0; i < startRow; i++)
		{
			getline(file, line);
		}

		while (getline(file, line))
		{
			stringstream ss(line);
			string valueStr;
			vector<double> dataRecord;
			for (unsigned int i = 0; i < inputColumns + outputColumns; i++)
			{
				getline(ss, valueStr, ',');
				dataRecord.push_back(stod(valueStr));
			}
			this->dataSet.push_back(dataRecord);
		}

		file.close();
	}
}

void NeuralNetwork::setNormalizationValues(double inputMin, double inputMax, double outputMin, double outputMax)
{
	this->minInput = inputMin;
	this->minOutput = outputMin;
	this->maxInput = inputMax;
	this->maxOutput = outputMax;
}

void NeuralNetwork::setValuesToInputLayer(vector<double> &inputValues)
{
	this->layers[0].setInputValues(inputValues);
}

void NeuralNetwork::setValuesToOutputLayer(vector<double> &outputValues)
{
	this->layers[this->layers.size() - 1].setOutputValues(outputValues);
}

void NeuralNetwork::shuffleDataSet()
{
	shuffle(this->dataSet.begin(), this->dataSet.end(), default_random_engine(11));
}

void NeuralNetwork::train(unsigned int epochs)
{
	vector<unsigned int> trainIndices;
	unsigned int maxTrainIndex = static_cast<unsigned int>(this->dataSet.size() * this->trainPercentage);
	for (unsigned int i = 0; i < maxTrainIndex; i++)
	{
		trainIndices.push_back(i);
	}

	const unsigned int outputLayerSize = this->layers[this->layers.size() - 1].size();
	for (unsigned int i = 0; i < epochs; i++)
	{
		vector<double> accumErrors(outputLayerSize, 0);
		for (unsigned int j = 0; j < trainIndices.size(); j++)
		{
			unsigned int trainIndex = trainIndices[j];
			setValuesToInputLayer(getInputsFromDataRecord(this->dataSet[trainIndex]));
			setValuesToOutputLayer(getOutputsFromDataRecord(this->dataSet[trainIndex]));
			feedForward();
			backPropagation();
			accumulateStepErrors(accumErrors);
		}
		cout << "Epoch " << i << ": " << getEpochError(accumErrors, trainIndices.size()) << endl;
		shuffle(trainIndices.begin(), trainIndices.end(), default_random_engine(NULL));
	}
	cout << endl;
}
