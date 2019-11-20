#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

namespace ImportModelExample {
	int main()
	{
		const string MODEL_FILE = "..\\ExportedModel.txt";
		auto network = NeuralNetwork::importModel(MODEL_FILE);
		auto outputs = network.predict(vector<double>{ 510, 1571 });
		for (unsigned int i = 0; i < outputs.size(); i++)
			cout << outputs[i] << " ";
		cout << endl;

		cout << "----------------WEIGHTS------------------" << endl;
		network.printWeights();

		cin.get();
		return 0;
	}
}