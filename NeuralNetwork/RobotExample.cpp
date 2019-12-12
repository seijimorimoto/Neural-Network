#define NOMINMAX
#include <iostream>
#include <limits>
#include <cmath>
#include <ctime>
#include "aria.h"
#include "NeuralNetwork.h"

using namespace std;

namespace RobotExample {
	// Gets the output values of the record in a training dataset that is closest to given inputs.
	// Params:
	// - trainDataSet: Vector containing train records. Each train record is represented also as a
	// vector, but of double precision values.
	// - inputs: A vector with the inputs that are going to be used to find the closest record.
	// Returns: The output values of the record in the dataset that is closest to given inputs.
	vector<double> getClosestTrainingRecordOutputs(vector<vector<double>> &trainDataSet, vector<double> &inputs)
	{
		// Setting the initial min distance of a record to the given inputs to a max value.
		double minDistance = numeric_limits<double>::max();
		unsigned int minIndex = -1;

		// Iterate over the dataset.
		for (unsigned int i = 0; i < trainDataSet.size(); i++)
		{
			// Calculate the similarity of a record to the given inputs by using the euclidean
			// distance between the given inputs and the input values of the record.
			double distance = 0;
			for (unsigned int j = 0; j < inputs.size(); j++)
			{
				double attrDif = trainDataSet[i][j] - inputs[j];
				distance += attrDif * attrDif;
			}
			distance = sqrt(distance);

			// Update the min distance to the calculated distance and keep the index of the
			// record used.
			if (distance < minDistance)
			{
				minDistance = distance;
				minIndex = i;
			}
		}

		// Get only the output values from the training record that was found to be closest to the
		// given inputs.
		vector<double> outputs;
		for (unsigned int i = inputs.size(); i < trainDataSet[minIndex].size(); i++)
		{
			outputs.push_back(trainDataSet[minIndex][i]);
		}

		return outputs;
	}
	
	int main(int argc, char *argv[])
	{
		// Constants for importing the model and testing it.
		const string MODEL_FILE = "ExportedModel.txt";
		const int NUM_INPUTS = 2;
		const int NUM_OUTPUTS = 2;
		const int TEST_TIME = 120;
		const string TRAIN_FILE = "RobotDataFiltered.csv";
		
		// Importing the neural network and getting the data (in the appropriate format) that will
		// be used for calculating test errors.
		auto network = NeuralNetwork::importModel(MODEL_FILE);
		network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, 1, 1, 0);
		auto trainDataSet = network.getDataSet();

		// Initializing variables used for the testing process.
		vector<double> errors(NUM_OUTPUTS, 0);
		double totalError = 0;
		unsigned int n = 0;
		clock_t beginTime = clock();

		// Create instances.
		Aria::init();
		ArRobot robot;

		// Parse command line arguments.
		ArArgumentParser argParser(&argc, argv);
		argParser.loadDefaultArguments();

		// Connect to the robot.
		ArRobotConnector robotConnector(&argParser, &robot);
		if (robotConnector.connectRobot()) cout << "Robot connected!" << endl;

		// Enable the motors in the robot.
		robot.runAsync(false);
		robot.lock();
		robot.enableMotors();
		robot.unlock();

		// Vector for storing the readings of the robot sensors.
		ArSensorReading *sonarSensors[8];

		// The main loop of the program. It uses the sensor readings as inputs to the neural
		// network in order to get values for its left and right wheel speed as outputs. It ends
		// when a certain amount of time (TEST_TIME) has passed.
		do
		{
			// Initialize the distance to the front and left side of the robot to something
			// above sensor max values.
			double frontDistance = 10000;
			double leftDistance = 10000;

			// Get the readings from each sonar sensor of the robot.
			for (unsigned int i = 0; i < 8; i++)
			{
				sonarSensors[i] = robot.getSonarReading(i);
			}

			// Assign the minimum reading of the first 3 sensors (left) as the left distance.
			for (unsigned int i = 0; i < 3; i++)
			{
				if (sonarSensors[i]->getRange() < leftDistance)
					leftDistance = sonarSensors[i]->getRange();
			}

			// Assign the minimum reading of the 4th and 5th sensor as the front distance.
			for (unsigned int i = 3; i < 5; i++)
			{
				if (sonarSensors[i]->getRange() < frontDistance)
					frontDistance = sonarSensors[i]->getRange();
			}

			// Use the neural network to predict speed values based on the left and front distance.
			vector<double> inputs{ leftDistance, frontDistance };
			auto outputs = network.predict(inputs);
			
			// Get the values that were expected to be obtained from the neural network by looking
			// at the values of the closest record in the imported data set (from TRAIN_FILE).
			auto expectedOutputs = getClosestTrainingRecordOutputs(trainDataSet, inputs);

			// Accumulate the square of the error of each output given by the neural network.
			// The error is calculated by substracting the output of the network from the expected
			// output.
			for (unsigned int i = 0; i < NUM_OUTPUTS; i++)
			{
				auto outputError = expectedOutputs[i] - outputs[i];
				errors[i] += outputError * outputError;
			}
			
			// Increment the number of times the neural network has been used to predict so far.
			n++;

			// Set the output values of the neural network to the left and right wheel speed of the
			// robot. Then wait a little bit to start the next iteration of the loop.
			robot.setVel2(outputs[0], outputs[1]);
			ArUtil::sleep(100);
		} while (double(clock() - beginTime) / CLOCKS_PER_SEC < TEST_TIME);

		// Calculate the root-mean-squared-error of each output neuron and accumulate those results
		// in another variable (totalError).
		for (unsigned int i = 0; i < NUM_OUTPUTS; i++)
		{
			errors[i] /= n;
			errors[i] = sqrt(errors[i]);
			totalError += errors[i];
			cout << "OUTPUT NEURON " << i << " ERROR: " << errors[i] << endl;
		}

		// Find the total error of the test process (i.e. the mean of the root-mean-squared-error
		// of the output neurons).
		totalError /= 2;
		cout << "TOTAL OUTPUT ERROR: " << totalError << endl;

		// Stop the robot
		robot.lock();
		robot.stop();
		robot.unlock();

		// Wait for user input to be able to see results in the console before it closes itself.
		cin.get();
		
		// Terminate all threads and exit
		Aria::exit();

		return 0;
	}
}