#include "Examples.h"
#include <iostream>
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
}