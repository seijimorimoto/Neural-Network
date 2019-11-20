#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

namespace ImportModelExample {
	int main()
	{
		const string MODEL_FILE = "..\\ExportedModel.txt";
		auto network = NeuralNetwork::importModel(MODEL_FILE);

		cout << "----------------WEIGHTS------------------" << endl;
		network.printWeights();

		cin.get();
		return 0;
	}
}