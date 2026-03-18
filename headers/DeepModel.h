
/**
 * @file DeepModel.h
 * @brief Main interface for the data analysis and training of neural networks, which is based on the linear algebra interface. 
 *
 * It gives the user the opportunity to use the classes Dataset, Matrix and Neuralnetwork.
 */



#pragma once

#include "activation.h"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>

/**
 * @brief Enables reading datasets from .csv files and doing various operations on them.
 * 
 * Note :   The input and output layer (last and first layer) of the choosen neuralnetwork 
 *          need to matcht the dataset matrix shapes. 
 * @see Neuralnetwork
 */
class Dataset
{

public:
    /** @brief Input data of the choosen Dataset, represented as stacked matrix of shape d1xd2xd3, where d3 is equal to the sample size.  */
    Matrix input;

    /** @brief Input data of the choosen Dataset, represented as stacked matrix of shape d1xd2xd3, where d3 is equal to the sample size.  */
    Matrix expected;
    
    /** @brief Creates a empty dataset. */
    Dataset();

    /** 
     * @brief   Reads a dataset from a .csv file, ignoring malformed rows and string-valued columns.              
     * 
     * @param filename Relative path to the file from the current working directory.
     * @param label Column index of the expected output. (default = 0)
     * */
    Dataset(const std::string filename, size_t label_col = 0);


    /** 
     * @brief   Reads a dataset from a .csv file, ignoring malformed rows and string-valued columns.
     *                  
     * @param filename Relative path to the file from the current working directory.
     * @param label Column index of the expected output.
     * */
    Dataset(const std::string filename, const std::vector<size_t>& ignore, size_t label_col = 0);



    /**
     * @brief   Splits the dataset in two parts.
     * 
     * @param ratio Proportion of the data assigned to the first set. (between 0 and 1)
     * @return Pair of (first set, second set)
     */
    std::pair<Dataset, Dataset> split(float ratio);


    /**
     * @brief Encodes the expected output into one hot vectors.
     * Automatically determines the number of unique values and changes the scalar output to the corresponding vectors.
     * 
     * @note Expected data must have shape 1x1xh.
     */
    void one_hot_encode();   


    /**
     * @brief Normalizes input data into range between 0 and 1.
     */
    void normalize ();


    /**
     *  @brief Standardizes the input data using z-score normalization, in which every feature as zero mean and unit variance.
     *  z = (x - μ) / σ
     */
    void standardize();

    /**
     * @brief Prints sample size, input dimension and output dimension.
     * 
     */
    void print_information();

    /** @brief Amount of samples */
    size_t sample_size();

    /** 
     * @brief Rows of one input input vector sample 
     * @note This is the input layer dimension for your neuralnetwork.
     * */
    size_t input_dim();

    /** 
     * @brief Rows of the expected vector 
     * @note This is the output layer dimension for your neuralnetwork.
     * */
    size_t expected_dim();

};



/**
 * @brief Build a neuralnetwork.
 * 
 * Note :   The input and output layer (last and first layer) of the choosen neuralnetwork 
 *          need to match witht he choosen dataset matrix shapes. 
 * 
 * @see Dataset
 */

class NeuralNetwork 
{   
private:

    static std::mt19937 gen;


    bool imported = false;
    bool print = true;
    
    Loss loss_function_class;

    size_t lfunc_type;
    std::vector<size_t> afunc_type;

    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_fn lfunc;
    loss_derivative_fn lfunc_dx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_fn> afunc;
    std::vector<activation_fn> afunc_dx;

    std::vector<Matrix> weight_matrices;
    std::vector<Matrix> bias_matrices;


    void gradient_descent(const size_t steps, Dataset &ds, double lr, double lambda, size_t batch_size);
    void print_status(const size_t step, const size_t steps, const size_t batch_size, const size_t dataset_size, size_t& current_epoch);
    std::vector<Matrix> layer_outputs(const Matrix& input);
    

public:

    /** @brief Creates an empty neuralnetwork.  */
    NeuralNetwork();

    /** @brief Disables all following print messages and status displays. */
    void disable_print();


    /**
     * @brief Adds another layer of the neuralnetwork.
     * 
     * @param neurons Number of neurons
     * @param atype Activation function for this layer (example : Activation::ReLU)
     *  
     * @see Activation
     */
    void add_layer(const size_t neurons, activation_type  atype);


    /**
     * @brief Sets the loss function .
     * @note Softmax works only as activation function, if you use cross entropy here.
     * 
     * @param lytpe Loss function (example : Loss::QUADRATIC)
     * 
     * @see Loss
     */
    void configure_loss_function(loss_type ltype);



    /**
     * @brief Configures the input layer.
     * 
     * @param neurons Number of neurons
     * @note Use ds.input_dim() as neurons to match the input layer dimension to your dataset ds.
     * 
     *  */    
    void configure_input_layer(const size_t neurons);


    /**
     * @brief Sets the loss weights for any loss function.
     * 
     * @param w  vector of floats of size equal to the amount of output layer neurons
     * @note This works for every loss function.
     * 
     *  */  
    void set_loss_weights(const std::vector<float> w);
    

    /**
     * @brief initalises random weights 
     * 
     * @param begin start range
     * @param end end range
     * 
     * @note If you're using no parameters, the default range is (-0.1, 0.1)
     * 
     */
    void initalise_random_weights(float begin = -0.1, float end = 0.1);

    /** @brief Initalises the weights with the xavier initalisation. */
    void initalise_xavier_weights();

     /** @brief Initalises the weights with the He initalisation. */
    void initalise_he_weights();



    /**
     * @brief Runs the backpropagation algorithm to minimize the loss. 
     * 
     * @param epochs Epochs
     * @param Dataset The trainingset, which needs be compatible for the input & output-layer of this network.
     * @param ofunc The choosen optimizing algorithm (exmaple : Optimizer::MIN_BATCH_GRADIENT_DESCENT)
     * @param batch_size The batch_size for each step.
     * 
     * @see Optimizer
     */
    void fit(const size_t epochs,Dataset &ds, optimizer_type ofunc, double lr, size_t batch_size = 64);


    /**
     * @brief Runs the backpropagation algorithm to minimize the loss, while using advanced parameters to
     * enable L2 regulazation.
     * 
     * @param epochs Epochs
     * @param Dataset The trainingset, which needs be compatible for the input & output-layer of this network.
     * @param ofunc The choosen optimizing algorithm (exmaple : Optimizer::MIN_BATCH_GRADIENT_DESCENT)
     * @param param A instance of the class Hyperparameter, which contains all necessary information for the algorithm.
     * 
     * example:
     * 
     *      Hyperparameter r;
     *      r.batch_size = 64;
     *      r.lambda = 10e-8     // <- L2 enables
     *      r.lr = 0.001;
     *      fit(10, ds, Optimizer::MIN_BATCH_GRADIENT_DESCENT, r);
     *  @see Optimizer
     */
    void fit(const size_t epochs,Dataset &ds, optimizer_type ofunc, Hyperparameter &param);


    
    
    /**
     * @brief Runs the backpropagation algorithm with Adam / AdamW.
     * 
     * @param epochs Epochs
     * @param Dataset The trainingset, which needs be compatible for the input & output-layer of this network.
     * @param adam   A instance of the class Hyperparameter, which contains all necessary information for the adam.
     * 
     *      ADAM_Optimizer adam;
     *      adam.batch_size = 64;
     *      adam.lambda = 10e-8     // <- AdamW enabled
     *      adam.lr = 0.001;
     *      (You can also configure beta1, beta2 etc..)
     *      fit(10, ds, adam);
     * 
     * @see ADAM_Optimizer
     */
    void fit(const size_t epochs,Dataset &ds, ADAM_Optimizer &adam);


    /**
     * @brief Runs the network for a stacked input matrix.
     * 
     * @param input matrix of shape ILD x 1 x H, where ILD is the input layer dimension and H the amount of samples.
     * 
     * @return Matrix of shape OLD x 1 x H, where OLD is the output layer dimension-
     * 
     */
    Matrix run(const Matrix& input);



    /**
     * @brief Calculates the accuracy of a classificator neuralnetwork. Works only if the Dataset is one hot encoded.
     * @param Dataset The dataset
     * 
     * @note You dont need to use the function .one_hot_encode of the Dataset class. Any one hot encoded expected values are okay.
     */
    float accuracy(Dataset &ds);


    /**
     * @brief Calculates the accuracy for a classificator neuralnetwork and prints it.
     * 
     * @param Dataset The Dataset
     * @param name Alias for the dataset to increase readability (example : "Dataset1")
     * 
     * @note There is also a version without the name parameter.
     */
    void performance(Dataset& ds, std::string name);

    /**
     * @brief Calculates the accuracy for a classificator neuralnetwork and prints it.
     * 
     * @param Dataset The Dataset
     */
    void performance(Dataset& ds);
    
    void binary_confusion_matrix(Dataset& ds, const float threshold = 0.5);


    /**
     * @brief Loads all weights and activations functions  of a txt, which was prior created with .save_weights()
     * 
     */
    void load_weights(const std::string &filename);


    /**
     * @brief Saves all weights and activations functions  to a .txt.
     * 
     * After that you can reuse the network with load_weights()
     * 
     */
    void save_weights(const std::string &filename);

    /**
     * @brief Sets the seed for this class and matrix.
     * 
     */
    static void set_seed(size_t seed);
};








