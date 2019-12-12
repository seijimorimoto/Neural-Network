#include "Examples.h"
#include <string>

using namespace std;

int main(int argc, char * argv[])
{
	// Default case is the RobotExample.
	if (argc == 1)
	{
		return RobotExample::main(argc, argv);
	}
	
	string argument = argv[1];

	if (argument == "ExpectedBehaviourTestExample")
	{
		return ExpectedBehaviourTestExample::main();
	}

	if (argument == "ImportModelExample")
	{
		return ImportModelExample::main();
	}
	
	if (argument == "MultipleTrainersExample")
	{
		return MultipleTrainersExample::main();
	}

	if (argument == "RobotExample")
	{
		return RobotExample::main(argc, argv);
	}
	
	if (argument == "TrainExample")
	{
		return TrainExample::main();
	}

	if (argument == "ValidateExample")
	{
		return ValidateExample::main();
	}
}