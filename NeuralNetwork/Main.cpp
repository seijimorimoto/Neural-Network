#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

int main()
{
	const unsigned int EPOCHS = 5;
	const string FILE_PATH = "..\\RobotDataFiltered.csv";
	const double LAMBDA = 0.2;
	const double LEARNING_RATE = 0.6;
	const double MIN_INPUT = 0;
	const double MIN_OUTPUT = 0;
	const double MAX_INPUT = 5000;
	const double MAX_OUTPUT = 300;
	const double MOMENTUM = 0.1;
	const unsigned int NUM_INPUTS = 2;
	const unsigned int NUM_OUTPUTS = 2;
	const unsigned int START_ROW = 1;
	const double TRAIN = 1;
	const double VALIDATE = 0;
	
	vector<NeuronLayer> layers;
	layers.push_back(NeuronLayer(2, LAMBDA));
	layers.push_back(NeuronLayer(4, LAMBDA));
	layers.push_back(NeuronLayer(2, LAMBDA));
	
	NeuralNetwork network(layers, LEARNING_RATE, MOMENTUM);
	network.initializeWeights();
	network.setCsvDataFile(FILE_PATH, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
	network.shuffleDataSet();

	/*
	cout << "-----------------DATA SET----------------" << endl;
	network.printDataSet();
	*/

	//cout << "-----------NORMALIZED DATA SET-----------" << endl;
	network.normalizeDataSet(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);
	//network.printDataSet();
	
	cout << "-----------------ERRORS------------------" << endl;
	network.train(EPOCHS);
	
	/*
	cout << "------------ACTIVATION VALUES------------" << endl;
	network.printActivationValues();
	
	cout << "-------------LOCAL GRADIENTS-------------" << endl;
	network.printLocalGradients();

	cout << "------------WEIGHTS UPDATED--------------" << endl;
	network.printWeights();
	*/

	cin.get();
	return 0;
}