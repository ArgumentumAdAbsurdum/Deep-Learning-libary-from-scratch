




cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark
