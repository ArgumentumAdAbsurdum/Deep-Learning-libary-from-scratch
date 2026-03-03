#pragma once
#include "model.h"
#include "activationCPU.h"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>



template<>
class dataset<CPU>
{

public:
    std::vector<matrix<CPU>> input;
    std::vector<matrix<CPU>> expected;
    
    dataset();
    dataset(const std::string filename, size_t label_col = 0);
    dataset(const std::string filename, const std::vector<size_t> label_cols);
    dataset(const std::string filename, const std::vector<size_t> input_cols, const std::vector<size_t> label_cols);

    void one_hot_encode();      // hier prüfen ob dim von matrix<CPU> bei expected immer 1x1 ist.
    void normalize ();
    void standardize();

};


// Das kommt weg. 
template<>
class model<CPU>
{
protected:
    
    size_t lfunc_type;
    std::vector<size_t> afunc_type;


    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_fn lfunc;
    loss_derivative_fn lfunc_dx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_fn> afunc;
    std::vector<activation_fn> afunc_dx;

    model();

public:

    void add_layer(const size_t neurons, activation_type  atype);
    void configure_loss_function(loss_type ltype);
    void configure_input_layer(const size_t neurons);





};






template<>
class classificator<CPU> : public model<CPU>
{   
private:
    std::vector<matrix<CPU>> weight_matrices;

    void mini_batch_gradient_descent(const size_t epochs, dataset<CPU> &ds);
    void batch_gradient_descent(const size_t epochs, dataset<CPU> &ds);
    void stochastic_gradient_descent(const size_t epochs, dataset<CPU> &ds, double& lr);
    
    std::vector<matrix<CPU>> layer_outputs(const matrix<CPU>& input);
    matrix<CPU> run(const matrix<CPU>& input);

public:

    classificator<CPU>();
    dataset<CPU> load_csv(const std::string& filename, size_t label_col = 0); // das zu dataset?


    void initalise();
    void initalise_random(float begin, float end);
    void initalise_xavier();
    void initalise_he();

    void fit(const size_t epochs,dataset<CPU> &ds, optimizer_type ofunc, double lr = 0.1);
    void fit(const size_t epochs,dataset<CPU> &ds, adam_optimizer<CPU> &adam);

    void performance(dataset<CPU>& ds, std::string name);
    void performance(dataset<CPU>& ds);


    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename);
};
