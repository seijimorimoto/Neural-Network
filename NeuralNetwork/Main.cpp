#include "Examples.h"
#include <string>

using namespace std;

int main(int argc, char * argv[])
{
	if (argc == 1)
	{
		return TrainExample::main();
	}

	string argument = argv[1];
	
	if (argc == 2 && argument == "ImportModelExample")
	{
		return ImportModelExample::main();
	}

	if (argc == 2 && argument == "RobotExample")
	{
		return RobotExample::main(argc, argv);
	}
}