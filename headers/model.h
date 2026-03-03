#pragma once
#include "backend.h"



template<typename Backend>
class model;

template<typename Backend>
class dataset;

template<typename Backend>
class classificator;



#ifdef ENABLE_CUDA
    #include "modelCUDA.cu"
    using Classificator = classificator<CUDA>;

#else
    #include "modelCPU.h"
    using Classificator = classificator<CPU>;
    using Dataset = dataset<CPU>;
#endif