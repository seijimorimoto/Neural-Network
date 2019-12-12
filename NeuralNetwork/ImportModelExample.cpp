#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

namespace ImportModelExample {
	int main()
	{
		// Import a neural network model from a file.
		const string MODEL_FILE = "ExportedModel.txt";
		auto network = NeuralNetwork::importModel(MODEL_FILE);

		// Get a prediction from the neural network by giving it some input values.
		auto outputs = network.predict(vector<double>{ 510, 1571 });

		// Print the output values obtained as a prediction of the neural network.
		for (unsigned int i = 0; i < outputs.size(); i++)
			cout << outputs[i] << " ";
		cout << endl;

		// Print the weights of the imported model.
		cout << "----------------WEIGHTS------------------" << endl;
		network.printWeights();

		// Wait for user input to be able to see results in the console before it closes itself.
		cin.get();
		return 0;
	}
}