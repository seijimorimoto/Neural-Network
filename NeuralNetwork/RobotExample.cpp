#include "aria.h"
#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

namespace RobotExample {
	int main(int argc, char *argv[])
	{
		const string MODEL_FILE = "ExportedModel.txt";
		auto network = NeuralNetwork::importModel(MODEL_FILE);

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
		while (true)
		{
			double frontDistance = 0;
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
				frontDistance += sonarSensors[i]->getRange();
			}
			frontDistance /= 2;
			
			auto outputs = network.predict(vector<double>{ leftDistance, frontDistance });
			for (unsigned int i = 0; i < outputs.size(); i++)
				cout << outputs[i] << " ";
			cout << endl;

			robot.setVel2(outputs[0], outputs[1]);
			ArUtil::sleep(100);
		}

		// Stop the robot
		robot.lock();
		robot.stop();
		robot.unlock();

		// Terminate all threads and exit
		Aria::exit();
		return 0;
	}
}