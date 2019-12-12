#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

namespace ExpectedBehaviourTestExample {
	int main()
	{
		const int EPOCHS = 1;
		const string MODEL_FILE = "ExpectedBehaviourInitialModel.txt";
		const string TRAIN_FILE = "ExpectedBehaviourTrainData.csv";
		auto network = NeuralNetwork::importModel(MODEL_FILE);
		network.setCsvDataFile(TRAIN_FILE, 2, 2, 0, 1, 0);
		network.normalizeDataSet();
		network.train(EPOCHS);

		cout << "---------------DATA SET------------------" << endl;
		network.printDataSet();

		cout << "-----------ACTIVATION VALUES-------------" << endl;
		network.printActivationValues();

		cout << "------------LOCAL GRADIENTS--------------" << endl;
		network.printLocalGradients();

		cout << "----------------WEIGHTS------------------" << endl;
		network.printWeights();

		cin.get();
		return 0;
	}
}