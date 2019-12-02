#define NOMINMAX
#include <iostream>
#include <limits>
#include <cmath>
#include <ctime>
#include "aria.h"
#include "NeuralNetwork.h"

using namespace std;

namespace RobotExample {
	vector<double> getClosestTrainingRecordOutputs(vector<vector<double>> &trainDataSet, vector<double> &inputs)
	{
		double minDistance = numeric_limits<double>::max();
		unsigned int minIndex = -1;

		for (unsigned int i = 0; i < trainDataSet.size(); i++)
		{
			double distance = 0;
			for (unsigned int j = 0; j < inputs.size(); j++)
			{
				double attrDif = trainDataSet[i][j] - inputs[j];
				distance += attrDif * attrDif;
			}
			distance = sqrt(distance);
			if (distance < minDistance)
			{
				minDistance = distance;
				minIndex = i;
			}
		}

		vector<double> outputs;
		for (unsigned int i = inputs.size(); i < trainDataSet[minIndex].size(); i++)
		{
			outputs.push_back(trainDataSet[minIndex][i]);
		}

		return outputs;
	}
	
	int main(int argc, char *argv[])
	{
		const string MODEL_FILE = "ExportedModel.txt";
		const int NUM_INPUTS = 2;
		const int NUM_OUTPUTS = 2;
		const int TEST_TIME = 180;
		const string TRAIN_FILE = "RobotDataFiltered.csv";
		
		auto network = NeuralNetwork::importModel(MODEL_FILE);
		network.setCsvDataFile(TRAIN_FILE, NUM_INPUTS, NUM_OUTPUTS, 1, 1, 0);
		auto trainDataSet = network.getDataSet();
		vector<double> errors(NUM_OUTPUTS, 0);
		double totalError = 0;
		unsigned int n = 0;
		clock_t beginTime = clock();

		// Create instances
		Aria::init();
		ArRobot robot;

		// Parse command line arguments
		ArArgumentParser argParser(&argc, argv);
		argParser.loadDefaultArguments();

		ArRobotConnector robotConnector(&argParser, &robot);
		if (robotConnector.connectRobot()) cout << "Robot connected!" << endl;

		robot.runAsync(false);
		robot.lock();
		robot.enableMotors();
		robot.unlock();

		ArSensorReading *sonarSensors[8];
		do
		{
			double frontDistance = 10000;
			double leftDistance = 10000;
			for (unsigned int i = 0; i < 8; i++)
			{
				sonarSensors[i] = robot.getSonarReading(i);
			}

			for (unsigned int i = 0; i < 3; i++)
			{
				if (sonarSensors[i]->getRange() < leftDistance)
					leftDistance = sonarSensors[i]->getRange();
			}

			for (unsigned int i = 3; i < 5; i++)
			{
				if (sonarSensors[i]->getRange() < frontDistance)
					frontDistance = sonarSensors[i]->getRange();
			}

			vector<double> inputs{ leftDistance, frontDistance };
			auto outputs = network.predict(inputs);
			auto expectedOutputs = getClosestTrainingRecordOutputs(trainDataSet, inputs);

			for (unsigned int i = 0; i < NUM_OUTPUTS; i++)
			{
				auto outputError = expectedOutputs[i] - outputs[i];
				errors[i] += outputError * outputError;
			}
			
			n++;

			robot.setVel2(outputs[0], outputs[1]);
			ArUtil::sleep(100);
		} while (double(clock() - beginTime) / CLOCKS_PER_SEC < TEST_TIME);

		for (unsigned int i = 0; i < NUM_OUTPUTS; i++)
		{
			errors[i] /= n;
			errors[i] = sqrt(errors[i]);
			totalError += errors[i];
			cout << "OUTPUT NEURON " << i << " ERROR: " << errors[i] << endl;
		}
		totalError /= 2;
		cout << "TOTAL OUTPUT ERROR: " << totalError << endl;

		// Stop the robot
		robot.lock();
		robot.stop();
		robot.unlock();

		cin.get();
		
		// Terminate all threads and exit
		Aria::exit();

		return 0;
	}
}