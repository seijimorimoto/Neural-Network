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
	string parse(double value, int precision = 1)
	{
		stringstream ss;
		ss << std::fixed << std::setprecision(precision) << value;
		return ss.str();
	}

	int main()
	{
		const unsigned int EPOCHS = 100;
		const double MIN_INPUT = 0;
		const double MIN_OUTPUT = 0;
		const double MAX_INPUT = 5000;
		const double MAX_OUTPUT = 300;
		const unsigned int NUM_INPUTS = 2;
		const unsigned int NUM_OUTPUTS = 2;
		const string BASE_OUT_FILE = "ExportedModels/ExportedModel";
		const unsigned int START_ROW = 1;
		const double TRAIN = 0.7;
		const string TRAIN_FILE = "RobotDataFiltered.csv";
		const double VALIDATE = 0.3;

		const vector<int> HIDDEN_NEURONS{ 4, 6, 8, 10 };
		const vector<double> LAMBDAS{ 0.2, 0.4, 0.6, 0.8 };
		const vector<double> MOMENTUMS{ 0.2, 0.4, 0.6, 0.8 };
		const vector<double> LEARNING_RATES{ 0.2, 0.4, 0.6, 0.8 };
		
		map<string, double> modelToErrorMapping;

		for (unsigned int i = 0; i < HIDDEN_NEURONS.size(); i++)
		{
			auto hiddenNeurons = HIDDEN_NEURONS[i];
			for (unsigned int j = 0; j < LAMBDAS.size(); j++)
			{
				auto lambda = LAMBDAS[j];
				for (unsigned int k = 0; k < MOMENTUMS.size(); k++)
				{
					auto momentum = MOMENTUMS[k];
					for (unsigned int l = 0; l < LEARNING_RATES.size(); l++)
					{
						auto learningRate = LEARNING_RATES[l];
						string modelName = to_string(hiddenNeurons) + "_" + parse(lambda) + "_"
							+ parse(momentum) + "_" + parse(learningRate);
						
						vector<NeuronLayer> layers;
						layers.push_back(NeuronLayer(2, lambda));
						layers.push_back(NeuronLayer(hiddenNeurons, lambda));
						layers.push_back(NeuronLayer(2, lambda));
						
						NeuralNetwork network(layers, learningRate, momentum);
						network.initializeWeights();
						network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, START_ROW, TRAIN, VALIDATE);
						network.shuffleDataSet();
						network.setNormalizationValues(MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT);
						network.normalizeDataSet();
						network.train(EPOCHS);

						modelToErrorMapping[modelName] = network.validate();
						network.exportModel(BASE_OUT_FILE + "_" + modelName + ".txt");
						cout << modelName << ": " << modelToErrorMapping[modelName] << endl;
					}
				}
			}
		}
		cout << endl;

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

		cout << "The model with the smallest error was: " << minErrorModel << endl;
		cout << "The error of the model was: " << minErrorValue << endl;

		cin.get();
		return 0;
	}
}