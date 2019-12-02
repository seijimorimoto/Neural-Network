#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <limits>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include "NeuralNetwork.h"

// Constructs a neural network.
// Params:
// - layers: Vector of layers that will compose the neural network.
// - learningRate: Rate at which the network will learn (value from 0 to 1).
// - momentum: Value that expresses how much to consider previous updates when making a new update
//   to the network's weights (value from 0 to 1).
NeuralNetwork::NeuralNetwork(vector<NeuronLayer> &layers, double learningRate, double momentum)
{
	this->layers = layers;
	this->learningRate = learningRate;
	this->momentum = momentum;
}

// Default destructor of a neural network.
NeuralNetwork::~NeuralNetwork() = default;

// Adds the current errors (squared) of the output neurons to the history of errors (squared) of
// those neurons in an epoch.
// Params:
// - accumErrors: Vector with the size of the output layer. Each position within it represents a
// neuron in the output layer and holds the accumulated errors (squared) of an epoch for that
// neuron.
void NeuralNetwork::accumulateStepErrors(vector<double>& accumErrors)
{
	// Iterates over the 'accumErrors' vector and adds, to the corresponding position, the error
	// squared of the output neurons.
	for (unsigned int i = 0; i < accumErrors.size(); i++)
	{
		auto neuronError = this->layers[this->layers.size() - 1].getNeuronError(i);
		accumErrors[i] += neuronError * neuronError;
	}
}

// Performs the back-propagation process of the neural network.
void NeuralNetwork::backPropagation()
{
	// Compute the errors and the local gradients of the output layer neurons. These cannot be
	// computed along with the hidden layers, since the local gradient calculation is different.
	unsigned int i = this->layers.size() - 1;
	this->layers[i].computeErrors();
	this->layers[i].computeLocalGradients();

	// Iteratively (from the last to the first hidden layer) compute the local gradients of the
	// neurons at each hidden layer.
	for (i = i - 1; i > 0; i--)
	{
		vector<Neuron> *neurons = this->layers[i + 1].neurons;
		int biasNextLayer = this->layers[i + 1].biasN;
		this->layers[i].computeLocalGradients(neurons, biasNextLayer);
	}

	// Iteratively update the weights of the neurons at each layer (from the first hidden layer to
	// the output layer).
	for (i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *neurons = this->layers[i - 1].neurons;
		this->layers[i].updateWeights(neurons, this->learningRate, this->momentum);
	}
}

// Denormalizes a set of data by using the output normalization values set for the neural network.
// Params:
// - outputs: Vector of data to denormalize.
void NeuralNetwork::denormalizeOutputs(vector<double> &outputs)
{
	for (unsigned int i = 0; i < outputs.size(); i++)
	{
		outputs[i] = outputs[i] * (this->maxOutput - this->minOutput) + this->minOutput;
	}
}

// Exports a neural network model to a file.
// Params:
// - filePath: The path to the file where the neural network model will be exported to.
void NeuralNetwork::exportModel(string filePath)
{
	// Open the file to write.
	ofstream file;
	file.open(filePath);

	// If it was possible to open the file...
	if (file.is_open())
	{
		// Output to the file the information corresponding to each layer in the neural network.
		file << "LAYERS\n";
		for (unsigned int i = 0; i < this->layers.size(); i++)
		{
			auto neurons = this->layers[i].size() - this->layers[i].biasN;
			auto biasN = this->layers[i].biasN;
			auto lambda = this->layers[i].lambda;
			auto minWeight = this->layers[i].minWeight;
			auto maxWeight = this->layers[i].maxWeight;
			file << neurons << "," << biasN << "," << lambda << "," << minWeight << "," << maxWeight << "\n";
		}
		file << "\n";
		
		// Output to the file the information corresponding to the weights of each neuron in the neural
		// network.
		file << "WEIGHTS\n";
		for (unsigned int i = 1; i < this->layers.size(); i++)
		{
			auto neuronsWeights = this->layers[i].getWeights();
			for (unsigned int j = 0; j < neuronsWeights.size(); j++)
			{
				file << i << "," << j;
				for (unsigned int k = 0; k < neuronsWeights[j].size(); k++)
				{
					file << "," << neuronsWeights[j][k];
				}
				file << "\n";
			}
		}
		file << "\n";

		// Output to the file the information of the parameters used to train and normalize the neural
		// network. Then close the file.
		file << "PARAMS\n";
		file << "LEARNING_RATE," << this->learningRate << "\n";
		file << "MIN_INPUT," << this->minInput << "\n";
		file << "MIN_OUTPUT," << this->minOutput << "\n";
		file << "MAX_INPUT," << this->maxInput << "\n";
		file << "MAX_OUTPUT," << this->maxOutput << "\n";
		file << "MOMENTUM," << this->momentum << "\n";
		file.close();
	}
}

// Performs the feed-forward process of the neural network.
void NeuralNetwork::feedForward()
{
	// Starting from the first hidden layer, iteratively compute the input value and activation values
	// of the neurons in each layer.
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		vector<Neuron> *prevLayerNeurons = this->layers[i - 1].neurons;
		this->layers[i].computeInputs(prevLayerNeurons);
		this->layers[i].computeActivationValues();
	}
}

// Returns the dataset that was loaded into the neural network.
vector<vector<double>> NeuralNetwork::getDataSet()
{
	return this->dataSet;
}

// Imports a neural network model from a file.
// Params:
// - filePath: The path to the model.
// Returns: The neural network constructed from the imported model.
NeuralNetwork NeuralNetwork::importModel(string filePath)
{
	// Open the file for reading.
	ifstream file;
	file.open(filePath);

	// If it was possible to open the file...
	if (file.is_open())
	{
		double learningRate, minInput, minOutput, maxInput, maxOutput, momentum;
		vector<NeuronLayer> layers;
		string line;
		unsigned int phase = 0;
		
		// Read each line of the file until the end of file is reached.
		while (getline(file, line))
		{
			// Set the 'phase' variable based on the type of values we will be reading.
			if (line == "LAYERS")
			{
				phase = 0;
				continue;
			}
			else if (line == "WEIGHTS")
			{
				phase = 1;
				continue;
			}
			else if (line == "PARAMS")
			{
				phase = 2;
				continue;
			}
			else if (line == "")
				continue;

			stringstream ss(line);
			switch (phase)
			{
				case 0:
				{
					// Parse the information of the layer and then create the NeuronLayer object
					// with it.
					string neurons, biasN, lambda, minWeight, maxWeight;
					getline(ss, neurons, ',');
					getline(ss, biasN, ',');
					getline(ss, lambda, ',');
					getline(ss, minWeight, ',');
					getline(ss, maxWeight, ',');
					// TODO: Change type of minWeight and maxWeight to double.
					layers.push_back(NeuronLayer(stoi(neurons), stoi(biasN), stod(lambda), stod(minWeight), stod(maxWeight)));
					break;
				}
				case 1:
				{
					// Parse the information of the weights (the layer and the neuron we are
					// referring to in the file line), format it as a vector and then set it
					// in the corresponding layer-neuron we already created.
					string layer, neuron, weight;
					vector<double> *neuronWeights = new vector<double>();
					getline(ss, layer, ',');
					getline(ss, neuron, ',');
					while (getline(ss, weight, ','))
					{
						neuronWeights->push_back(stod(weight));
					}
					layers[stoi(layer)].setWeight(neuronWeights, stoi(neuron));
					break;
				}
				case 2:
				{
					// Parse the information of the parameters used to train and normalize the
					// neural network.
					string paramLabel, paramValue;
					getline(ss, paramLabel, ',');
					getline(ss, paramValue, ',');
					if (paramLabel == "LEARNING_RATE")
						learningRate = stod(paramValue);
					else if (paramLabel == "MIN_INPUT")
						minInput = stod(paramValue);
					else if (paramLabel == "MIN_OUTPUT")
						minOutput = stod(paramValue);
					else if (paramLabel == "MAX_INPUT")
						maxInput = stod(paramValue);
					else if (paramLabel == "MAX_OUTPUT")
						maxOutput = stod(paramValue);
					else if (paramLabel == "MOMENTUM")
						momentum = stod(paramValue);
					break;
				}
			}
		}

		// Close the file, create the neural network with the information that was parsed from the file
		// and return the network.
		file.close();
		NeuralNetwork network(layers, learningRate, momentum);
		network.setNormalizationValues(minInput, maxInput, minOutput, maxOutput);
		return network;
	}
}

// Initializes all the weights in the neural network.
void NeuralNetwork::initializeWeights()
{
	srand(static_cast<unsigned int>(time(nullptr)));
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		const unsigned int n = this->layers[i - 1].size();
		this->layers[i].initializeWeights(n);
	}
}

// Gets the error of an epoch (using the mean of root-mean-squared-error of each output neuron).
// Params:
// - stepErrors: Vector with the size of the output layer. Each position within it represents a
// neuron in the output layer and holds the accumulated errors (squared) of the epoch for that
// neuron.
// - epochSize: Number of data records used in the epoch.
// Returns: The error of the epoch.
double NeuralNetwork::getEpochError(vector<double> &stepErrors, unsigned int epochSize)
{
	double epochError = 0;
	for (unsigned int i = 0; i < stepErrors.size(); i++)
	{
		epochError += sqrt(stepErrors[i] / epochSize);
	}
	epochError /= stepErrors.size();
	return epochError;
}

// Gets the values of the input features of a given data record.
// Params:
// - dataRecord: Vector of values representing a data record (following same structure as the
// records loaded in the 'dataset' attribute).
// Returns: A vector with only the values of the input features of the data record.
vector<double> NeuralNetwork::getInputsFromDataRecord(vector<double> &dataRecord)
{
	vector<double> inputs;
	for (unsigned int i = 0; i < this->inputFeatures; i++)
	{
		inputs.push_back(dataRecord[i]);
	}
	return inputs;
}

// Returns a vector with the output values of the network (i.e. the activation values of the output
// neurons).
vector<double> NeuralNetwork::getNetworkOutputs()
{
	return this->layers[this->layers.size() - 1].getActivationValues();
}

// Gets the values of the output features of a given data record.
// Params:
// - dataRecord: Vector of values representing a data record (following same structure as the
// records loaded in the 'dataset' attribute).
// Returns: A vector with only the values of the output features of the data record.
vector<double> NeuralNetwork::getOutputsFromDataRecord(vector<double> &dataRecord)
{
	vector<double> outputs;
	for (unsigned int i = this->inputFeatures; i < this->inputFeatures + this->outputFeatures; i++)
	{
		outputs.push_back(dataRecord[i]);
	}
	return outputs;
}

// Returns a vector with the indices of the dataset used for training.
vector<unsigned int> NeuralNetwork::getTrainIndices()
{
	vector<unsigned int> trainIndices;
	unsigned int maxTrainIndex = static_cast<unsigned int>(this->dataSet.size() * this->trainPercentage);
	for (unsigned int i = 0; i < maxTrainIndex; i++)
	{
		trainIndices.push_back(i);
	}
	return trainIndices;
}

// Normalizes the loaded dataset with the normalization values set for the neural network.
void NeuralNetwork::normalizeDataSet()
{
	// Iterate over the dataset.
	for (unsigned int i = 0; i < this->dataSet.size(); i++)
	{
		// Iterate over the input features in the data record and normalize them.
		for (unsigned int j = 0; j < this->inputFeatures; j++)
		{
			this->dataSet[i][j] = (this->dataSet[i][j] - this->minInput) / (this->maxInput - this->minInput);
		}

		// Iterate over the output features in the data record and normalize them.
		for (unsigned int j = this->inputFeatures; j < this->inputFeatures + this->outputFeatures; j++)
		{
			this->dataSet[i][j] = (this->dataSet[i][j] - this->minOutput) / (this->maxOutput - this->minOutput);
		}
	}
}

// Normalizes a set of data by using the input normalization values set for the neural network.
// Params:
// - inputs: Vector of data to normalize.
void NeuralNetwork::normalizeInputs(vector<double>& inputs)
{
	for (unsigned int i = 0; i < inputs.size(); i++)
	{
		inputs[i] = (inputs[i] - this->minInput) / (this->maxInput - this->minInput);
	}
}

// Gives a prediction of output values based on given inputs.
// Params:
// - inputs: Vector with the value of each input feature. From this vector, we make the prediction.
// Returns: A prediction of the output values.
vector<double> NeuralNetwork::predict(vector<double> &inputs)
{
	normalizeInputs(inputs);
	setValuesToInputLayer(inputs);
	feedForward();
	auto outputs = getNetworkOutputs();
	denormalizeOutputs(outputs);
	return outputs;
}

// Prints the activation values of the neurons at each layer of the neural network.
void NeuralNetwork::printActivationValues()
{
	for (unsigned int i = 0; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printActivationValues();
		cout << endl;
	}
	cout << endl;
}

// Prints the dataset loaded for the neural network.
void NeuralNetwork::printDataSet()
{
	for (unsigned int i = 0; i < this->dataSet.size(); i++)
	{
		for (unsigned int j = 0; j < this->inputFeatures; j++)
		{
			cout << this->dataSet[i][j] << " ";
		}
		cout << " | ";
		for (unsigned int j = this->inputFeatures; j < this->inputFeatures + this->outputFeatures; j++)
		{
			cout << " " << this->dataSet[i][j];
		}
		cout << endl;
	}
	cout << endl;
}

// Prints the local gradients of the neurons at each hidden and output layer of the neural network.
void NeuralNetwork::printLocalGradients()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printLocalGradients();
		cout << endl;
	}
	cout << endl;
}

// Prints all the weights in the neural network.
void NeuralNetwork::printWeights()
{
	for (unsigned int i = 1; i < this->layers.size(); i++)
	{
		cout << "NEURON LAYER " << i << endl;
		this->layers[i].printWeights();
		cout << endl;
	}
	cout << endl;
}

// Loads a dataset from a CSV file into the 'dataSet' attribute of the NeuralNetwork object.
// Params:
// - csvFilePath: Path to the CSV file.
// - inputColumns: Number of columns in the CSV file that correspond to input values.
// - outputColumns: Number of columns in the CSV file that correspond to output values.
// - startRow: Row in the CSV file where the values of the dataset start appearing (rows are
// indexed starting in 0).
// - trainPerc: Percentage of the dataset to be used for training of the neural network.
// - validationPerc: Percentage of the dataset to be used for validation of the neural network.
void NeuralNetwork::setCsvDataFile(string csvFilePath, unsigned int inputColumns, unsigned int outputColumns, unsigned int startRow, double trainPerc, double validationPerc)
{
	// Open CSV file for reading.
	ifstream file;
	file.open(csvFilePath);

	// Set attributes in the object based on parameters passed in the function call.
	this->inputFeatures = inputColumns;
	this->outputFeatures = outputColumns;
	this->trainPercentage = trainPerc;
	this->validationPercentage = validationPerc;

	// If it was possible to open the file...
	if (file.is_open())
	{
		// Skip lines until we get to the starting row of the dataset values.
		string line;
		for (unsigned int i = 0; i < startRow; i++)
		{
			getline(file, line);
		}

		// Read each line of the file until the end of the file.
		// Get the values from each line and format them as a vector. Then push it to the dataset.
		while (getline(file, line))
		{
			stringstream ss(line);
			string valueStr;
			vector<double> dataRecord;
			for (unsigned int i = 0; i < inputColumns + outputColumns; i++)
			{
				getline(ss, valueStr, ',');
				dataRecord.push_back(stod(valueStr));
			}
			this->dataSet.push_back(dataRecord);
		}

		// Close the file.
		file.close();
	}
}

// Sets key values in the neural network used for normalizing and denormalizing data.
// Params:
// - inputMin: Minimum value that input features can take.
// - inputMax: Maximum value that input features can take.
// - outputMin: Minimum value that ouputs can take.
// - outputMax: Maximum value that outputs can take.
void NeuralNetwork::setNormalizationValues(double inputMin, double inputMax, double outputMin, double outputMax)
{
	this->minInput = inputMin;
	this->minOutput = outputMin;
	this->maxInput = inputMax;
	this->maxOutput = outputMax;
}

// Sets the input values of the neurons in the input layer.
// Params:
// - inputValues: Vector of input values to be set (in order) in the input neurons.
void NeuralNetwork::setValuesToInputLayer(vector<double> &inputValues)
{
	this->layers[0].setInputValues(inputValues);
}

// Sets the expected values of the neurons in the output layer.
// Params:
// - outputValues: Vector of expected values to be set (in order) in the output neurons.
void NeuralNetwork::setValuesToOutputLayer(vector<double> &outputValues)
{
	this->layers[this->layers.size() - 1].setOutputValues(outputValues);
}

// Shuffles the dataset loaded for the neural network.
void NeuralNetwork::shuffleDataSet()
{
	shuffle(this->dataSet.begin(), this->dataSet.end(), default_random_engine(11));
}

// Trains a neural network for a number of epochs.
// Params:
// - epochs: Number of times the whole training set will be used by the neural network to learn.
// - printEpochErrors: True if the error at each epoch should be printed. False otherwise.
void NeuralNetwork::train(unsigned int epochs, bool printEpochErrors)
{
	auto trainIndices = getTrainIndices();
	const auto outputLayerSize = this->layers[this->layers.size() - 1].size();

	// Iterate over the number of epochs.
	for (unsigned int i = 0; i < epochs; i++)
	{
		// Train the network.
		vector<double> accumErrors(outputLayerSize, 0);
		trainEpoch(trainIndices, accumErrors);

		// Print the train errors if the 'printEpochErrors' flag is on.
		if (printEpochErrors)
		{
			cout << "EPOCH " << i << ": " << endl;
			cout << "  Train: " << getEpochError(accumErrors, trainIndices.size()) << endl;
		}

		// Shuffle the training indices, so that in next epoch the network sees the train records
		// in different order.
		shuffle(trainIndices.begin(), trainIndices.end(), default_random_engine(NULL));
	}

}

// Trains a neural network for a number of epochs or when an early stopping criteria is met.
// Params:
// - epochs: Number of times the whole training set will be used by the neural network to learn.
// - minDelta: Minimum decrease in error between epochs for it to be considered an improvement in
// accuracy of the neural network.
// - patience: Number of epochs without accuracy improvement that should pass for the neural
// network to abort training.
// - printEpochErrors: True if the error at each epoch should be printed. False otherwise.
void NeuralNetwork::train(unsigned int epochs, double minDelta, unsigned int patience, bool printEpochErrors)
{
	auto trainIndices = getTrainIndices();
	const auto outputLayerSize = this->layers[this->layers.size() - 1].size();
	double minValidError = numeric_limits<double>::max();
	unsigned int epochsWithoutImprovement = 0;

	// Iterate over the number of epochs. Finish when having reached the number of epochs or when
	// having more than 'patience' consecutive epochs without improvement.
	for (unsigned int i = 0; i < epochs && epochsWithoutImprovement <= patience; i++)
	{
		// Train the network.
		vector<double> accumErrors(outputLayerSize, 0);
		trainEpoch(trainIndices, accumErrors);

		// Get the training and validation errors of the epoch.
		const double trainError = getEpochError(accumErrors, trainIndices.size());
		const double validError = validate();
		
		// When the validationError of this epoch is less than the minimum validation error seen in
		// previous epochs by at least a 'minDelta' amount, reset the number of epochs without
		// improvement to 0. Otherwise, increase it by 1.
		if (minValidError - validError >= minDelta)
		{
			epochsWithoutImprovement = 0;
		}
		else
		{
			epochsWithoutImprovement++;
		}

		// Update the minimum validation error seen (if the current validation error is smaller
		// than what was the minimum so far).
		minValidError = min(minValidError, validError);
		
		// Print the train and validation errors if the 'printEpochErrors' flag is on.
		if (printEpochErrors)
		{
			cout << "EPOCH " << i << ": " << endl;
			cout << "  Train: " << trainError << endl;
			cout << "  Validation: " << validError << endl;
		}

		// Shuffle the training indices, so that in next epoch the network sees the train records
		// in different order.
		shuffle(trainIndices.begin(), trainIndices.end(), default_random_engine(NULL));
	}
	
	if (printEpochErrors)
	{ 
		cout << endl;
	}
}

// Trains a neural network for a single epoch.
// Params:
// - trainIndices: The indices within the dataset that correspond to the training set.
// - accumErrors: Vector with the size of the output layer. Each position within the vector should
// be 0 before calling this method. After completion of this method, each position will contain
// the sum of the squares of the errors of that neuron for the epoch.
void NeuralNetwork::trainEpoch(vector<unsigned int> &trainIndices, vector<double>& accumErrors)
{
	// Iterate over the indices of the training set.
	for (unsigned int i = 0; i < trainIndices.size(); i++)
	{
		// Assign the input and expected output of the training record to the input and output
		// layers respectively. Then perform the feed-forward back-propagation process and
		// gather the errors from each output neuron.
		unsigned int trainIndex = trainIndices[i];
		setValuesToInputLayer(getInputsFromDataRecord(this->dataSet[trainIndex]));
		setValuesToOutputLayer(getOutputsFromDataRecord(this->dataSet[trainIndex]));
		feedForward();
		backPropagation();
		accumulateStepErrors(accumErrors);
	}
}

// Validates the performance of a neural network against a validation set.
// Returns: The error of executing the neural network against the validation set.
double NeuralNetwork::validate()
{
	// Get the start and ending position of the validation set within the dataset.
	unsigned int startValidationIndex = static_cast<unsigned int>(this->dataSet.size() * this->trainPercentage);
	unsigned int endValidationIndex = static_cast<unsigned int>(this->dataSet.size() * (this->trainPercentage + this->validationPercentage));
	const unsigned int outputLayerSize = this->layers[this->layers.size() - 1].size();
	vector<double> neuronErrors(outputLayerSize, 0);

	// Iterate over the indices of the validation set.
	for (unsigned int i = startValidationIndex; i < endValidationIndex; i++)
	{
		// Assign the input and expected output of the validation record to the input and output
		// layers respectively. Then perform the feed-forward process and calculate the errors at
		// output layer.
		setValuesToInputLayer(getInputsFromDataRecord(this->dataSet[i]));
		setValuesToOutputLayer(getOutputsFromDataRecord(this->dataSet[i]));
		feedForward();
		this->layers[this->layers.size() - 1].computeErrors();
		accumulateStepErrors(neuronErrors);
	}

	// Returns the overall error of the validation.
	return getEpochError(neuronErrors, endValidationIndex - startValidationIndex);
}
