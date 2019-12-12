#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

namespace ExpectedBehaviourTestExample {
	int main()
	{
		// Define constants for training a neural network model.
		const int EPOCHS = 1;
		const string MODEL_FILE = "ExpectedBehaviourInitialModel.txt";
		const string TRAIN_FILE = "ExpectedBehaviourTrainData.csv";

		// Import the neural network model and set the data to be used for training.
		auto network = NeuralNetwork::importModel(MODEL_FILE);
		network.setCsvDataFile(TRAIN_FILE, 2, 2, 0, 1, 0);
		network.normalizeDataSet();

		// Train the neural network.
		network.train(EPOCHS);

		// Print the dataset used to train the model.
		cout << "---------------DATA SET------------------" << endl;
		network.printDataSet();

		// Print the activation values of each neuron in the model.
		cout << "-----------ACTIVATION VALUES-------------" << endl;
		network.printActivationValues();

		// Print the local gradients of the neurons in the model.
		cout << "------------LOCAL GRADIENTS--------------" << endl;
		network.printLocalGradients();

		// Print the all the weights between neurons in the model.
		cout << "----------------WEIGHTS------------------" << endl;
		network.printWeights();

		// Wait for user input to be able to see results in the console before it closes itself.
		cin.get();
		return 0;
	}
}