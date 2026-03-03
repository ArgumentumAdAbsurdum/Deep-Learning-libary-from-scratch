#include "modelCPU.h"


model<CPU>::model() : input_layer_neurons(0)
{

}

void model<CPU>::add_layer(const size_t neurons, activation_type atype) 
{
    neurons_per_layer.push_back(neurons);
    afunc_type.push_back(atype);
    this->output_layer_neurons = neurons;
}

void model<CPU>::configure_loss_function(loss_type _ltype) 
{
    this->lfunc_type = _ltype;
}

void model<CPU>::configure_input_layer(const size_t neurons)
{
    this->input_layer_neurons = neurons;
    neurons_per_layer.insert(neurons_per_layer.begin(), neurons);
}






