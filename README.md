# Multiple-DNN-Scheduling

This repository includes the source code used in the project
["Distributed Inference for Multiple DNN Models in IoT Environments"](https://dl.acm.org/doi/abs/10.1145/3492866.3561254).


## About the project

Please refer to the [Poster-MobiHoc2022.pdf](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/Poster-MobiHoc2022.pdf)

The scheduling algorithms used in this project was written with reference to the following papers.
*  [PSO-GA and Genetic algorithm (GA)](https://github.com/SPSO-GA/dataset) - Chen, X., Zhang, J., Lin, B., Chen, Z., Wolter, K., & Min, G. (2021). Energy-efficient offloading for DNN-based smart IoT systems in cloud-edge environments. *IEEE Transactions on Parallel and Distributed Systems*, 33(3), 683-697.
*  [HEFT and CPOP](https://en.wikipedia.org/wiki/Heterogeneous_earliest_finish_time) - Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002). Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE transactions on parallel and distributed systems*, 13(3), 260-274.
*  [PEFT](https://github.com/mackncheesiest/peft) - Arabnejad, H., & Barbosa, J. G. (2013). List scheduling algorithm for heterogeneous systems by an optimistic cost table. *IEEE Transactions on Parallel and Distributed Systems*, 25(3), 682-694.

Most of the code is written in python, but the high-complexity functions such as calculating DAG completion times are written in C++ for optimization.


## File Description


*  [algorithms/ServerOrderEvolutionary.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/algorithms/ServerOrderEvolutionary.py) - This is an implementation of a layer-based scheduling algorithm including GA and PSO-GA using the server-order encoding strategy of the aforementioned paper.
*  [algorithms/ServerEvolutionary.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/algorithms/ServerEvolutionary.py) - Our proposed block-based memetic algorithm.
*  [algorithms/Greedy.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/algorithms/Greedy.py) - This code implements greedy scheduling algorithms, including HEFT, CPOP, and PEFT.
*  [auto_test.sh](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/auto_test.sh) - Bash Shell Script for Auto-Run Tests. You can set up a test environment by adjusting the arguments.
*  [build_script.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/build_script.py) - Build script to build cpplib.cpp and load it as a python library.
*  [config.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/config.py) - Config file containing information of IoT servers and DNN model structures.
*  [cpplib.cpp](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/cpplib.cpp) - C++ library for calculating DAG completion time. It calculates the same function about 200 times faster than the Python code implemented using for loop.
*  [dag_data_generator.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/dag_data_generator.py) - This is the code that generates the DNN workflow. It includes implementation of piecwise partitioning, so you can specify the granularity of the DNN structure. Multilevel graph partitioning also applies within this code. Therefore, this code finally returns the DNN block structure.
*  [dag_env.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/dag_env.py) - Deprecated.
*  [dag_server.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/dag_server.py) - ALL components for this project is in here.

Most of the code is written in python, but the high-complexity functions such as calculating DAG completion times are written in C++ for optimization.


## Prerequisites
#### General
*  python>=3.6
*  numpy
*  matplotlib

#### Scheduling Algorithm
*  Multicore CPU (at least 8 cores for evolutionary algorithms)  
*  GPU and Pytorch (for reinforcement learning algorithms)  

#### Data Generator
*  Pre-generated DNN model and IoT device data ([config.py](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/config.py))


## How to run
#### 1. Build C++ functions
>  python3 build_script.py build_ext --inplace

#### 2. Run the test script
>  sh ./auto_test.sh  

It takes a few seconds for heuristic algorithms, and 1 to 5 minutes for evolutionary algorithms.  
The results are created in the ./outputs/ folder.  