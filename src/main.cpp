#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>





int main()
{
    
    /*
    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(256, Activation::RELU);
    c.add_layer(128, Activation::RELU);
    c.add_layer(10, Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise_random_weights();

    
    Dataset train = Dataset("../datasets/mnist_train.csv");
    Dataset test = Dataset("../datasets/mnist_test.csv");
 
    train.normalize();
    test.normalize();

    train.one_hot_encode();
    test.one_hot_encode();
    
    c.fit(1, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , 1);

    c.performance(train);
    c.performance(test);

    c.save_weights("test1.txt");
    */


    Dataset train("../datasets/give_me_some_credit.csv", {0}, 1);
    Dataset test = train.split(0.8);

    train.standardize();
    test.standardize();

    train.one_hot_encode();
    test.one_hot_encode();


    NeuralNetwork net;
    net.add_layer(10, Activation::ELU);
    net.add_layer(64, Activation::ELU);
    net.add_layer(32, Activation::ELU);
    net.add_layer(2,  Activation::SOFTMAX);
    net.configure_loss_function(Loss::CROSS_ENTROPY);


    net.set_loss_weights({0.07, 0.93});
    net.initalise_random_weights();
    

    //net.fit(5, train, Optimizer::STOCHASTIC_GRADIENT_DESCENT, 0.01);


    
    ADAM_Optimizer adam;
    adam.lambda = 0;
    adam.lr = 0.0005;
    adam.batch_size = 4;
    net.fit(10, train, adam);
    

    net.binary_confusion_matrix(train, 0.2);

    std::cout << "---------------" << std::endl;
    net.binary_confusion_matrix(test, 0.25);
    net.binary_confusion_matrix(test, 0.20);
    net.binary_confusion_matrix(test, 0.15);


    net.save_weights("test1.txt");




}