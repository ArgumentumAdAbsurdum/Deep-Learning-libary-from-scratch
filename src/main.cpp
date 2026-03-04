#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{

    // TODO : model und classificator vereinen
    // classificator umbennen 
    // dataset fertig, error checking nach fit verschieben!
    // Adam optimizer fit
    // fit fertig machen -> Batch GD, mini Batch GD
    // L2 weight opt
    // save + load
    // conv layer
    // cuda
    
    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(128, Activation::RELU);
    c.add_layer(64, Activation::RELU);
    c.add_layer(10, Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise();

    
    Dataset train = Dataset("../datasets/mnist_train.csv");
    Dataset test = Dataset("../datasets/mnist_train.csv");
 
    train.normalize();
    test.normalize();

    train.one_hot_encode();
    test.one_hot_encode();
    
    c.fit(2, train, Optimizer::BATCH_GRADIENT_DESCENT, 0.001);

    c.performance(train);
    c.performance(test);


    
}