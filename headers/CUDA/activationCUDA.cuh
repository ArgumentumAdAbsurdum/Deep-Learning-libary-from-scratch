#pragma once
#include "matrixCUDA.cuh"
#include <functional>
#include "activation.h"



/**
 * @brief Implementation of all Activation functions in CUDA.
 * 
 * Functions : 
 * Activation::IDENTITY
 * Activation::RELU         =
 * Activation::ELU          
 * Activation::SIGMOID      
 * Activation::LOG_SIGMOID  
 * Activation::HARD_SIGMOID 
 * Activation::TANH         
 * Activation::SOFTMAX      
 * 
 * 
 */
template<>
class activation<CUDA>
{   
private:
    static matrix<CUDA> ones(const matrix<CUDA>& a);
    static matrix<CUDA> identity(const matrix<CUDA>& a);
    static matrix<CUDA> relu(const matrix<CUDA>& a);
    static matrix<CUDA> elu(const matrix<CUDA>& a);
    static matrix<CUDA> sigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> log_sigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> hard_sigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> tanh(const matrix<CUDA>& a);
    static matrix<CUDA> softmax(const matrix<CUDA>& a);


    static matrix<CUDA> didentity(const matrix<CUDA>& a);
    static matrix<CUDA> drelu(const matrix<CUDA>& a);
    static matrix<CUDA> delu(const matrix<CUDA>& a);
    static matrix<CUDA> dsigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> dlog_sigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> dhard_sigmoid(const matrix<CUDA>& a);
    static matrix<CUDA> dtanh(const matrix<CUDA>& a);

   inline static float ELU_ALPHA_PARAM = 1;

public:

    static activation_type IDENTITY     = 0;
    static activation_type RELU         = 1;
    static activation_type ELU          = 2;
    static activation_type SIGMOID      = 3;
    static activation_type LOG_SIGMOID  = 4;
    static activation_type HARD_SIGMOID = 5;
    static activation_type TANH         = 6;
    static activation_type SOFTMAX      = 7;

    static activation_fn get_fn(activation_type atype);
    static activation_fn get_derivative_fn(activation_type atype);
};


/**
 * @brief Implementation of all loss functions in CUDA.
 * 
 * Functions : 
 * Loss::QUADRATIC
 * Loss::CROSS_ENTROPY
 */
template<>
class loss<CUDA>
{  
private:


    float cross_entropy(const matrix<CUDA> &expected, const matrix<CUDA> &result);
    float quadratic(const matrix<CUDA> &expected, const matrix<CUDA> &result);

    matrix<CUDA> dcross_entropy(const matrix<CUDA> &probability, const matrix<CUDA> &expected);
    matrix<CUDA> dcross_entropy_inkl_softmax(const matrix<CUDA> &probability, const matrix<CUDA> &expected);
    matrix<CUDA> dquadratic(const matrix<CUDA> &probability, const matrix<CUDA> &expected);

public:

    matrix<CUDA> weights;
    loss<CUDA>();
    static loss_type CROSS_ENTROPY = 0;
    static loss_type QUADRATIC = 1;


    loss_fn  get_fn(loss_type ltype);
    loss_derivative_fn get_derivative_fn(loss_type ltype, activation_type atype);
};


/**
 * @brief Implementation of basic optimizer types in CUDA.
 * 
 * Functions : 
 * Optimizer::STOCHASTIC_GRADIENT_DESCENT
 * Optimizer::BATCH_GRADIENT_DESCENT
 * Optimizer::MIN_BATCH_GRADIENT_DESCENT
 */
template<>
class optimizer<CUDA>
{
public:
    static optimizer_type STOCHASTIC_GRADIENT_DESCENT = 0;
    static optimizer_type BATCH_GRADIENT_DESCENT = 1;
    static optimizer_type MIN_BATCH_GRADIENT_DESCENT = 2;
};



/**
 * @brief Adam optimizer
 * 
 * @param lr learning rate (default = 0.001) 
 * @param beta1 (default = 0.9)
 * @param beta2 (default = 0.99)
 * @param epsilon (default = 10e-8)
 * @param batch_size (default = 64)
 * @param lambda Parameter for l2 regulazation. (default 10e-4)
 * 
 * example:
 * 
 * ADAM_Optimizer adam:
 * adam.lr = 0.1;
 * 
 */
template<>
class adam_optimizer<CUDA> : private optimizer<CUDA>
{
public:
    adam_optimizer(); 
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double lambda;
    size_t batch_size;
};

inline adam_optimizer<CUDA>::adam_optimizer() : lr(0.001), beta1(0.9), beta2(0.999), epsilon(10e-8), lambda(10e-4), batch_size(64)
{}


/**
 * @brief Class for setting specialized hyperparameters.
 * 
 * @param lr learning rate (default = 0.001)
 * @param lambda Parameter for l2 regulazation. (default 10e-4)
 * @param batch_size (default = 64)
 */
template<>
class hyperparameter<CUDA> : private optimizer<CUDA>
{
public:
    hyperparameter(); 
    double lr;
    double lambda;
    size_t batch_size;
};

inline hyperparameter<CUDA>::hyperparameter() : lr(0.001), lambda(10e-4), batch_size(64)
{}


