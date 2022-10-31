# Multiple-DNN-Scheduling

This repository includes the source code used in the project
["Distributed Inference for Multiple DNN Models in IoT Environments"](https://dl.acm.org/doi/abs/10.1145/3492866.3561254).


## About the project

Please refer to the [Poster-MobiHoc2022.pdf](https://github.com/yhjh5302/Multiple-DNN-Scheduling/blob/master/Poster-MobiHoc2022.pdf)

The scheduling algorithms used in this project was written with reference to the following papers.
*  [PSO-GA and Genetic algorithm]( https://github.com/SPSO-GA/dataset) - Chen, X., Zhang, J., Lin, B., Chen, Z., Wolter, K., & Min, G. (2021). Energy-efficient offloading for DNN-based smart IoT systems in cloud-edge environments. *IEEE Transactions on Parallel and Distributed Systems*, 33(3), 683-697.
*  [HEFT and CPOP](https://en.wikipedia.org/wiki/Heterogeneous_earliest_finish_time) - Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002). Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE transactions on parallel and distributed systems*, 13(3), 260-274.
*  [PEFT](https://github.com/mackncheesiest/peft) - Arabnejad, H., & Barbosa, J. G. (2013). List scheduling algorithm for heterogeneous systems by an optimistic cost table. *IEEE Transactions on Parallel and Distributed Systems*, 25(3), 682-694.

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
