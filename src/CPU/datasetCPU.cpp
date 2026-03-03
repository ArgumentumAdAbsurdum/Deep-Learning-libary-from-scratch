#include "model.h"
#include "modelCPU.h"



dataset<CPU>::dataset(){
    
}

dataset<CPU>::dataset(const std::string filename, size_t label_col)
{
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> label_cols)
{
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> input_cols, const std::vector<size_t> label_cols)
{
}

void dataset<CPU>::one_hot_encode()
{
}

void dataset<CPU>::normalize()
{
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for(matrix<CPU> vec : this->input )
    {
        float current_min = *std::min_element(vec.begin(), vec.end());
        float current_max = *std::max_element(vec.begin(), vec.end());

        max = std::max(current_max, max);
        min = std::min(current_min, min);
    }

    if(max == min)
        throw std::runtime_error("max values equal min values, which results in a divison by zero");

    for(matrix<CPU>& vec : this->input)
    {
        
        vec = vec - min;
        vec = vec * (1 / (max - min));
    }
    
    
}

void dataset<CPU>::standardize()
{
}
