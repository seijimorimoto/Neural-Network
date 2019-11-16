#include <ctime>
#include <fstream>
#include <iostream>
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


void NeuralNetwork::feedForward()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *prevLayerNeurons = this->layers[i - 1].neurons;
		this->layers[i].computeInputs(prevLayerNeurons);
		this->layers[i].computeActivationValues();
	}
}


void NeuralNetwork::initializeWeights()
{
	srand(time(nullptr));
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		const unsigned int n = this->layers[i - 1].size();
		this->layers[i].initializeWeights(n);
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
	for (unsigned int i = 0; i < this->inputData.size(); i++)
	{
		for (unsigned int j = 0; j < this->inputData[i].size(); j++)
		{
			cout << this->inputData[i][j] << " ";
		}
		cout << " | ";
		for (unsigned int j = 0; j < this->outputData[i].size(); j++)
		{
			cout << " " << this->outputData[i][j];
		}
		cout << endl;
	}
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

void NeuralNetwork::setCsvDataFile(string csvFilePath, unsigned int inputColumns, unsigned int outputColumns, unsigned int startRow)
{
	ifstream file;
	file.open(csvFilePath);

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
			vector<double> inputRecord, outputRecord;

			for (unsigned int i = 0; i < inputColumns; i++)
			{
				getline(ss, valueStr, ',');
				inputRecord.push_back(stod(valueStr));
			}

			for (unsigned int i = inputColumns; i < inputColumns + outputColumns; i++)
			{
				getline(ss, valueStr, ',');
				outputRecord.push_back(stod(valueStr));
			}

			this->inputData.push_back(inputRecord);
			this->outputData.push_back(outputRecord);
		}

		file.close();
	}
}

void NeuralNetwork::train(int epochs)
{
	for (unsigned int i = 0; i < epochs; i++)
	{
		feedForward();
		backPropagation();
	}
}
