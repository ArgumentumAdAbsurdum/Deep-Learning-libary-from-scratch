#include "DeepModel.h"
#include "modelCPU.h"
#include <algorithm>


dataset<CPU>::dataset()
{
    
}


dataset<CPU>::dataset(const std::string filename, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;

        float current_output = -1; 
        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell); 
            if (col == output_col) 
                current_output = static_cast<float>(value); 
            else 
                input_row.push_back(value); 
            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(1,1, current_output)); 
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
    
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> output_cols)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        std::vector<float> output_row;

        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell); 
            if (std::find(output_cols.begin(), output_cols.end(), col) != output_cols.end()) 
                output_row.push_back(static_cast<float>(value)); 
            else 
                input_row.push_back(value); 
            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(output_row.size(), 1, output_row));
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> input_cols, const std::vector<size_t> output_cols)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        std::vector<float> output_row;

        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell);

            if (std::find(output_cols.begin(), output_cols.end(), col) != output_cols.end()) 
                output_row.push_back(static_cast<float>(value)); 
            else if(std::find(input_cols.begin(), input_cols.end(), col) != input_cols.end()) 
                input_row.push_back(value); 

            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(output_row.size(), 1, output_row));
    }
    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
}


void dataset<CPU>::one_hot_encode()
{
    if(this->expected[0].columns() != 1 || this->expected[0].rows() != 1)
        throw std::runtime_error("one_hot_encode : Wrong matrix output shape for one hot encoding. It needs to be 1x1."); 
    

    
    std::vector<matrix<CPU>> res;
    res.reserve(this->expected.size());

    std::vector<float> values;
    values.reserve(this->expected.size());
    

    for(matrix<CPU> &mat : this->expected)
        values.push_back(mat[0]);
    
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    for(matrix<CPU> &mat : this->expected)
    {
        auto it = std::find(values.begin(), values.end(), mat[0]);
        int index = std::distance(values.begin(), it);

        matrix<CPU> _x = matrix<CPU>(values.size(), 1, 0);
        _x[index] = 1.0;
        res.push_back(_x);
    }

    this->expected = res;
    
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
        throw std::runtime_error("normalize : All values of the dataset are the same, which results in a divison by zero.");

    for(matrix<CPU>& vec : this->input)
    {
        
        vec = vec - min;
        vec = vec * (1 / (max - min));
    }
    
    
}


void dataset<CPU>::standardize()
{
    size_t rows = this->input[0].rows();
    std::vector<float> means(0);
    means.reserve(rows);

    std::vector<float> sigma(0);
    sigma.reserve(rows);


    for(matrix<CPU> &vec : this->input)
        for(size_t r = 0; r < rows; r++ )
            means[r] += vec[r]; 
         
    for(size_t r = 0; r < rows; r++ )
        means[r]  = means[r] / this->input.size();



    for(matrix<CPU> &vec : this->input)
        for(size_t r = 0; r < rows; r++ )
            sigma[r] += (vec[r] - means[r]) *(vec[r] - means[r]); 
    
    for(size_t r = 0; r < rows; r++ )
        sigma[r]  = std::sqrt(sigma[r] / this->input.size());



    for(matrix<CPU> &vec : this->input)
    {
        for(size_t r = 0; r < rows; r++ )
            vec[r] = (vec[r] - means[r]) / sigma[r];   
    }

}
