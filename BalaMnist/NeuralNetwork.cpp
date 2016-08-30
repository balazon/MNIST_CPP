
#include "NeuralNetwork.h"


void NeuralNetwork::addSimpleLayer(int nodeSize)
{
	int w = layers.size() == 0 ? inputSize + 1 : layers.back()->Theta.N() + 1;
	layers.push_back(std::make_shared<SimpleHiddenLayer>(nodeSize, w));
}

void NeuralNetwork::addOutputLayer(int outputSize)
{

}