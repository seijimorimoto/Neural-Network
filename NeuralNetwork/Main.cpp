#include <vector>
#include "NeuronLayer.h"
#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

int main()
{
	const double LAMBDA = 0.2;
	const double LEARNING_RATE = 0.6;
	const double MOMENTUM = 0.1;
	vector<NeuronLayer> layers;
	vector<double> inputs{ 1, 2 };
	vector<double> outputs{ 1, 0 };
	layers.push_back(NeuronLayer(2, LAMBDA, &inputs));
	layers.push_back(NeuronLayer(3, LAMBDA));
	layers.push_back(NeuronLayer(2, LAMBDA, nullptr, &outputs));
	layers[1].setWeight(new vector<double>{ 0.1, 0.2 }, 0);
	layers[1].setWeight(new vector<double>{ 0.3, 0.4 }, 1);
	layers[1].setWeight(new vector<double>{ 0.5, 0.6 }, 2);
	layers[2].setWeight(new vector<double>{ 0.5, 0.4, 0.3 }, 0);
	layers[2].setWeight(new vector<double>{ 0.2, 0.1, 0.6 }, 1);
	NeuralNetwork network(layers, LEARNING_RATE, MOMENTUM);
	
	network.feedForward();
	network.backPropagation();

	cout << "------------ACTIVATION VALUES------------" << endl;
	network.printActivationValues();
	
	cout << "-------------LOCAL GRADIENTS-------------" << endl;
	network.printLocalGradients();

	cout << "------------WEIGHTS UPDATED--------------" << endl;
	network.printWeights();

	cin.get();
	return 0;
}