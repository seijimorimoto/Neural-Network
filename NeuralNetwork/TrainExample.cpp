#include <iostream>
#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"

using namespace std;

namespace TrainExample
{
	int main()
	{
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

		vector<NeuronLayer> layers;
		layers.push_back(NeuronLayer(2, 1, LAMBDA));
		layers.push_back(NeuronLayer(HIDDEN_NEURONS, 1, LAMBDA));
		layers.push_back(NeuronLayer(2, 0, LAMBDA));

		NeuralNetwork network(layers, LEARNING_RATE, MOMENTUM);
		network.initializeWeights();
		network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
		network.shuffleDataSet();
		network.setNormalizationValues(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);
		network.normalizeDataSet();

		cout << "-----------------ERRORS------------------" << endl;
		network.train(EPOCHS, true);

		network.exportModel(OUT_FILE);

		cin.get();
		return 0;
	}
}