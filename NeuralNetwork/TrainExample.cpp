#include <iostream>
#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"

using namespace std;

namespace TrainExample
{
	int main()
	{
		// Defining constants for training the model.
		const unsigned int EPOCHS = 370;
		const int HIDDEN_NEURONS = 8;
		const double LAMBDA = 0.6;
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

		// Defining the structure of the multi-layer neural network.
		vector<NeuronLayer> layers;
		layers.push_back(NeuronLayer(2, 1, LAMBDA));
		layers.push_back(NeuronLayer(HIDDEN_NEURONS, 1, LAMBDA));
		layers.push_back(NeuronLayer(2, 0, LAMBDA));

		// Creating the neural network based on the configured layers. Also, setting up the data
		// for training the network and the initial (random) weights.
		NeuralNetwork network(layers, LEARNING_RATE, MOMENTUM);
		network.initializeWeights();
		network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
		network.shuffleDataSet();
		network.setNormalizationValues(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);
		network.normalizeDataSet();

		// Training the network and outputting the training error at each epoch.
		cout << "-----------------ERRORS------------------" << endl;
		network.train(EPOCHS, true);

		// Exporting the trained model to a file.
		network.exportModel(OUT_FILE);

		// Wait for user input to be able to see results in the console before it closes itself.
		cin.get();
		return 0;
	}
}