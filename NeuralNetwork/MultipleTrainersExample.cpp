#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"

using namespace std;

namespace MultipleTrainersExample
{
	// Parses a double precision value to a string.
	// Params:
	// - value: The double precision value to be parsed.
	// - precision: The number of decimal values to keep when parsing the double precision number.
	// Returns: The string representation of the double precision value.
	string parse(double value, int precision = 1)
	{
		stringstream ss;
		ss << std::fixed << std::setprecision(precision) << value;
		return ss.str();
	}

	int main()
	{
		// Defining constants for training multiple models.
		const unsigned int EPOCHS = 100;
		const double MIN_INPUT = 0;
		const double MIN_OUTPUT = 0;
		const double MAX_INPUT = 5000;
		const double MAX_OUTPUT = 300;
		const unsigned int NUM_INPUTS = 2;
		const unsigned int NUM_OUTPUTS = 2;
		const string BASE_OUT_FILE = "ExportedModels/ExportedModel";
		const unsigned int START_ROW = 1;
		const string SUMMARY_FILE = "SummaryModel.csv";
		const double TRAIN = 0.7;
		const string TRAIN_FILE = "RobotDataFiltered.csv";
		const double VALIDATE = 0.3;

		// Vectors containing different values for parameters and hyper-parameters of a neural
		// network. The models resulting from all combinations of these values are going to be
		// trained and the one with the smallest validation error will be indicated.
		const vector<int> HIDDEN_NEURONS{ 4, 6, 8, 10 };
		const vector<double> LAMBDAS{ 0.2, 0.4, 0.6, 0.8 };
		const vector<double> MOMENTUMS{ 0.2, 0.4, 0.6, 0.8 };
		const vector<double> LEARNING_RATES{ 0.2, 0.4, 0.6, 0.8 };
		
		map<string, double> modelToErrorMapping;
		ofstream summaryFile;
		summaryFile.open(SUMMARY_FILE);

		// Iterate over all values for the number of hidden neurons.
		for (unsigned int i = 0; i < HIDDEN_NEURONS.size(); i++)
		{
			auto hiddenNeurons = HIDDEN_NEURONS[i];
			// Iterate over all values for lambda.
			for (unsigned int j = 0; j < LAMBDAS.size(); j++)
			{
				auto lambda = LAMBDAS[j];
				// Iterate over all values for momentum.
				for (unsigned int k = 0; k < MOMENTUMS.size(); k++)
				{
					auto momentum = MOMENTUMS[k];
					// Iterate over all values of learning rate.
					for (unsigned int l = 0; l < LEARNING_RATES.size(); l++)
					{
						auto learningRate = LEARNING_RATES[l];
						// Get the name of model that is going to be trained in this iteration by
						// concatenating the values of the different parameters set for this
						// iteration.
						string modelName = to_string(hiddenNeurons) + "_" + parse(lambda) + "_"
							+ parse(momentum) + "_" + parse(learningRate);
						
						// Defining the structure of the multi-layer neural network.
						vector<NeuronLayer> layers;
						layers.push_back(NeuronLayer(2, 1, lambda));
						layers.push_back(NeuronLayer(hiddenNeurons, 1, lambda));
						layers.push_back(NeuronLayer(2, 0, lambda));
						
						// Creating the neural network based on the configured layers. Also,
						// setting up the data for training the network and the initial (random)
						// weights.
						NeuralNetwork network(layers, learningRate, momentum);
						network.initializeWeights();
						network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
						network.shuffleDataSet();
						network.setNormalizationValues(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);
						network.normalizeDataSet();

						// Training the model.
						network.train(EPOCHS);

						// Get the validation error and store it (associating it to the current
						// model). Then, export the model to a file and show the error in console,
						// as well as outputting it to another file.
						modelToErrorMapping[modelName] = network.validate();
						network.exportModel(BASE_OUT_FILE + "_" + modelName + ".txt");
						cout << modelName << ": " << modelToErrorMapping[modelName] << endl;
						summaryFile << modelName << "," << modelToErrorMapping[modelName] << "\n";
					}
				}
			}
		}
		cout << endl;
		summaryFile.close();

		// Find the model that had the minimum validation error.
		double minErrorValue = numeric_limits<double>::max();
		string minErrorModel = "";
		for (auto it = modelToErrorMapping.begin(); it != modelToErrorMapping.end(); ++it)
		{
			if (it->second < minErrorValue)
			{
				minErrorValue = it->second;
				minErrorModel = it->first;
			}
		}

		// Display the minimum validation error and the model associated with it.
		cout << "The model with the smallest error was: " << minErrorModel << endl;
		cout << "The error of the model was: " << minErrorValue << endl;

		// Wait for user input to be able to see results in the console before it closes itself.
		cin.get();
		return 0;
	}
}