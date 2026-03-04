#include "modelCPU.h"





neuralnetwork<CPU>::neuralnetwork() : input_layer_neurons(0)
{}

void neuralnetwork<CPU>::add_layer(const size_t neurons, activation_type atype) 
{
    neurons_per_layer.push_back(neurons);
    afunc_type.push_back(atype);
    this->output_layer_neurons = neurons;
}

void neuralnetwork<CPU>::configure_loss_function(loss_type _ltype) 
{
    this->lfunc_type = _ltype;
}

void neuralnetwork<CPU>::configure_input_layer(const size_t neurons)
{
    this->input_layer_neurons = neurons;
    neurons_per_layer.insert(neurons_per_layer.begin(), neurons);
}





// ------------------------------- BACKPROPAGATION -------------------------------------------

void neuralnetwork<CPU>::initalise()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i] + 1;      

        std::cout << "[LAYER = " << i << " WEIGHT MATRIX: => ROWS = " << rows << " , COLUMNS = " << cols << " ] " << std::endl;

        matrix<CPU> mat(rows, cols, -0.1, 0.1);    
        weight_matrices.push_back(mat);
    }



    lfunc = loss<CPU>::get_fn(lfunc_type);
    lfunc_dx = loss<CPU>::get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(activation<CPU>::get_fn(a));
        afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }


    if(afunc_type.back() == activation<CPU>::SOFTMAX && this->lfunc_type != loss<CPU>::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");

}


matrix<CPU> neuralnetwork<CPU>::run(const matrix<CPU> &input)
{
    matrix<CPU> result = input;
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {

        result.insert_row(0, 1);
        matrix<CPU> prod = weight_matrices[i] * result;
        
        result = afunc[i](prod);
    }

    return result;
}

std::vector<matrix<CPU>> neuralnetwork<CPU>::layer_outputs(const matrix<CPU> &input)
{
    std::vector<matrix<CPU>> outputs;
    outputs.resize(neurons_per_layer.size());
    outputs[0] = input;

    matrix<CPU> current = outputs[0];
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {

        current.insert_row(0, 1);
        matrix<CPU> prod = weight_matrices[i] * current;
        
        current = afunc[i](prod);

        outputs[i+1] = current;
    }

    return outputs;
}


void neuralnetwork<CPU>::mini_batch_gradient_descent(const size_t epochs, dataset<CPU>& ds, double lr , size_t batch_size)
{

}

void neuralnetwork<CPU>::batch_gradient_descent(const size_t epochs, dataset<CPU>& ds,  double lr)
{
    
    std::vector<matrix<CPU>> gradients;
    gradients.resize(weight_matrices.size());

    for(size_t ep = 0; ep < epochs; ep++)
    {
        for(size_t i = 0; i < weight_matrices.size(); i++)
            gradients[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);

        std::vector<matrix<CPU>> Z;
        std::vector<matrix<CPU>> Zb;
        
        Z.reserve(this->weight_matrices.size() + 1);
        Zb.reserve(this->weight_matrices.size() + 1);

        for(size_t pos = 0; pos < ds.input.size(); pos++)
        {
            matrix<CPU> &input_value = ds.input[pos];
            matrix<CPU> truth_value = ds.expected[pos];

            Z = layer_outputs(input_value);
            Zb = Z;

            for(matrix<CPU> &mat : Zb)
                mat.insert_row(0,1);


            ssize_t index = Z.size() - 1;
            matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Zb[index-1]); 

            gradients[index-1] += delta * Zb[index-1].transpose();
            
            
            index-= 2;
            for(index; index >= 0; index--)  
            {    
                matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
                weight_transposed.remove_row(0);
        
                delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Zb[index]);
                
                gradients[index] += delta * Zb[index].transpose();
            }
            
        }

        for(size_t i = 0; i < weight_matrices.size(); i++)
            weight_matrices[i] -= gradients[i] * (lr / ((float) ds.input.size())); 

    }
}

void neuralnetwork<CPU>::stochastic_gradient_descent(const size_t epochs, dataset<CPU>& ds, double lr)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,ds.input.size()-1); 

    for(size_t ep = 0; ep < epochs; ep++)
    {
        size_t pos = dist(rng);
        matrix<CPU> &input_value = ds.input[pos];
        matrix<CPU> truth_value = ds.expected[pos];

        std::vector<matrix<CPU>> Z = layer_outputs(input_value);
        std::vector<matrix<CPU>> Zb = Z;
        for(matrix<CPU> &mat : Zb)
            mat.insert_row(0,1);


        size_t index = Z.size() - 1;
        matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Zb[index-1]); 

        matrix<CPU> gradient = delta * Zb[index-1].transpose();

        weight_matrices[index-1] = weight_matrices[index-1] - lr * gradient;

    
        for(int i = index - 2; i >= 0; i--)  
        {    
            matrix<CPU> weight_transposed = weight_matrices[i+1].transpose();
            weight_transposed.remove_row(0);
        
            delta = (weight_transposed * delta) % afunc_dx[i](weight_matrices[i] * Zb[i]);
            gradient = delta * Zb[i].transpose();
            weight_matrices[i] = weight_matrices[i] - lr* gradient;
        }

    }


}









// ------------------------------------------------------------------------------------

void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU>& ds, optimizer_type ofunc, double lr, size_t batch_size )
{

    switch(ofunc)
    {
        case optimizer<CPU>::STOCHASTIC_GRADIENT_DESCENT:
            stochastic_gradient_descent(epochs, ds, lr);

        case optimizer<CPU>::BATCH_GRADIENT_DESCENT:
            batch_gradient_descent(epochs, ds, lr);

        case optimizer<CPU>::MIN_BATCH_GRADIENT_DESCENT:
            mini_batch_gradient_descent(epochs, ds, lr, batch_size);
    }
}



void neuralnetwork<CPU>::performance(dataset<CPU> &ds, std::string name)
{

    double rmsqe = 0;
    double accuracy = 0;
    for(size_t i = 0; i < ds.input.size(); i++)
    {
        matrix<CPU> pred = run(ds.input[i]);
        matrix<CPU> diff = (pred - ds.expected[i]);
        rmsqe += diff.L2();
        accuracy += (pred.argmax() == ds.expected[i].argmax());
    }   

    rmsqe = std::sqrt(rmsqe / ds.input.size());
    accuracy /= ds.input.size();

    std::cout << "[PERFORMANCE RESULTS FOR DATASET : " << name << "]" << std::endl;
    std::cout << "[ => Accuracy : " << accuracy << "]" <<  std::endl;
    std::cout << "[ => RMSQE : " << rmsqe << "]" << std::endl;

}

void neuralnetwork<CPU>::performance(dataset<CPU> &ds)
{
    performance(ds, "");
}
