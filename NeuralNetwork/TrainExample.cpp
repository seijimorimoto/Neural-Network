#include <iostream>
#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"

using namespace std;

namespace TrainExample
{
	int main()
	{
		const unsigned int EPOCHS = 150;
		const double LAMBDA = 0.8;
		const double LEARNING_RATE = 0.8;
		const double MIN_INPUT = 0;
		const double MIN_OUTPUT = 0;
		const double MAX_INPUT = 5000;
		const double MAX_OUTPUT = 300;
		const double MOMENTUM = 0.8;
		const unsigned int NUM_INPUTS = 2;
		const unsigned int NUM_OUTPUTS = 2;
		const string OUT_FILE = "ExportedModel.txt";
		const unsigned int START_ROW = 1;
		const double TRAIN = 1;
		const string TRAIN_FILE = "RobotDataFiltered.csv";
		const double VALIDATE = 0;

		vector<NeuronLayer> layers;
		layers.push_back(NeuronLayer(2, LAMBDA));
		layers.push_back(NeuronLayer(10, LAMBDA));
		layers.push_back(NeuronLayer(2, LAMBDA));

		NeuralNetwork network(layers, LEARNING_RATE, MOMENTUM);
		network.initializeWeights();
		network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
		network.shuffleDataSet();
		network.setNormalizationValues(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);

		/*
		cout << "-----------------DATA SET----------------" << endl;
		network.printDataSet();
		*/

		//cout << "-----------NORMALIZED DATA SET-----------" << endl;
		network.normalizeDataSet();
		//network.printDataSet();

		cout << "-----------------ERRORS------------------" << endl;
		network.train(EPOCHS, true);

		/*
		cout << "------------ACTIVATION VALUES------------" << endl;
		network.printActivationValues();

		cout << "-------------LOCAL GRADIENTS-------------" << endl;
		network.printLocalGradients();

		cout << "------------WEIGHTS UPDATED--------------" << endl;
		network.printWeights();
		*/

		network.exportModel(OUT_FILE);

		cin.get();
		return 0;
	}
}