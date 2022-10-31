# Multiple-DNN-Scheduling

This repository includes the source code used in the project
["Distributed Inference for Multiple DNN Models in IoT Environments"](https://dl.acm.org/doi/abs/10.1145/3492866.3561254).

## About the project

What is Piece-wise DNN Model Partitioning?

!['Submit New Job on the left -> Add New on the upper right'](DNN_model_structure.png "aaa")  

We split every DNN layer into several pieces, and consider the neuron level 
*  [analyzing-stream-modern-hardware](https://github.com/TU-Berlin-DIMA/analyzing-stream-modern-hardware) by TU-Berlin-DIMA
*  [LinearGenerator](https://github.com/walmartlabs/LinearGenerator) and [LinearRoad](https://github.com/walmartlabs/linearroad) by Walmart Labs

The algorithm used in this project was written with reference to the following papers.
*  [PSO-GA and Genetic algorithm]( https://github.com/SPSO-GA/dataset) - Chen, X., Zhang, J., Lin, B., Chen, Z., Wolter, K., & Min, G. (2021). Energy-efficient offloading for DNN-based smart IoT systems in cloud-edge environments. *IEEE Transactions on Parallel and Distributed Systems*, 33(3), 683-697.
*  [HEFT and CPOP](https://en.wikipedia.org/wiki/Heterogeneous_earliest_finish_time) - Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002). Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE transactions on parallel and distributed systems*, 13(3), 260-274.
*  [PEFT](https://github.com/mackncheesiest/peft) - Arabnejad, H., & Barbosa, J. G. (2013). List scheduling algorithm for heterogeneous systems by an optimistic cost table. *IEEE Transactions on Parallel and Distributed Systems*, 25(3), 682-694.

This project simulates an entire distributed inference process, from DNN model partitioning to evaluating .  
Most of the code is written in python, but the high-complexity tasks such as calculating DAG completion times are written in C++ for optimization.


## Prerequisites
#### General
*  Numpy
*  Pytorch
*  Pytorch

#### Data Generator
*  Plenty of free disk space (the generated data is about 1GB per lane!)

#### Data Server
*  ZeroMQ library
*  Cloud-Profiler library
*  Multicore CPU (at least 4 cores, and a core per every 4 Flink threads added)
*  Pre-generated data

#### Linear Road Benchmark
*  Apache Flink (WARNING: the Flink parallelism MUST be multiple of 4. Details are described below)
*  ZeroMQ library
*  Cloud-Profiler library
*  Multicore CPU (it is recommended to use the #cores as Flink parallelism. At least 4 is therefore required)

#### Validator
*  AeroSpike DB [server](https://www.aerospike.com/download/server). Client is already included in this project directory.
*  Pre-generated data
*  Output from Linear Road Benchmark


## How to run
#### 1. Configuring Necessary Libraries
Refer to [build-scripts](https://git.elc.cs.yonsei.ac.kr/bburg/build-scripts) to configure Cloud-Profiler and ZeroMQ libraries.  
After building Cloud-Profiler, locate the following files from `build_rel/src/cp/` directory:  
`cloud_profilerJNI.jar`, `libcloud_profiler.so`, `config_server/libnet_conf.so`  
Then copy the above files into /usr/lib/ directory. This may require sudo permission.  

*(optional)* If you're using Validator, install Aerospike DB server, and put client jar into JavaValidator directory.

#### 2. Building programs
Change directory to the project root folder before building each program!
###### 1. Data Generator
>  cd LinearGenerator  
>  mkdir com  
>  javac -d ./com ./src/*.java  

###### 2. Data Server
>  cd ZMQServer  
>  mvn clean  
>  mvn assembly:assembly -DdescriptorId=jar-with-dependencies

###### 3. Linear Road Benchmark
>  cd LRBFlinkZMQ  
>  mvn clean  
>  mvn assembly:assembly -DdescriptorId=jar-with-dependencies

###### 4. Validator
>  cd scripts  
>  ./compile_validator.sh

#### 3. Running the Benchmark
Change directory to the project root folder before running each program!
###### 1. Create Dataset
>  mkdir data  
>  cd scripts
>  ./create_dataset.sh ../data/*filename* *#-of-lanes*  

NOTE: the 'Total Number of notifications created' is necessary to run the server!  
If the program didn't run into any error, cardata file and tolldata file will be created.

###### 2. Launch Data Server
If you forgot the # of notifications, first run
>  wc -l data/*filename*  

and use it as the *number*
>  cd scripts  
>  ./run_server.sh ../data/*filename* *number* *cache-size*

When you see the message 'master port is open', the server is running and ready.

(About the cache size)  
The recommended cache size is 20000000 (20M). You can start by using this number first.  
If you get SIGSEGV, it is likely that the system does not have enough memory for such cache size.
In this case, reduce the size to 10M or less.  
If you have enough memory, you can increase the cache size for the sake of slower startup time, too.

###### 3. Launch Linear Road Benchmark
First turn on the Flink dashboard by running this in the Flink directory
>  bin/start-cluster.sh

Then, access *address-of-the-Flink-node*:8081 on the web browser. You'll see the Flink Dashboard interface.  
!['Submit New Job on the left -> Add New on the upper right'](%EC%A3%BC%EC%84%9D_2020-06-30_154357.png "aaa")  
As you upload the Linear Road Benchmark jar file we built before, the job will be added to the interface.  
!['Program Arguments: 0 4n server-address 0, Parallelism: 4n'](2020-06-30_154949.png "bbb")  
In the *Program Arguments*, type in necessary arguments as following:
*  arg0 must be 0
*  arg1 is number of threads we're using for this instance. IT MUST BE MULTIPLE OF 4, or the program will crash.
*  arg2 is address of the Data Server node. It must be currently running and ready server.
*  arg3 must be 0

And in the *Parallelism*, type in the same number as arg1.  
When the connection is successfully established, you'll see a message from Server like "New connection to port 6556 established" and the benchmark will run.

(About the # of threads)  
This is in fact due to an implementation fault. The server won't start new data sender thread unless it receives four connection request.  
What this means is that once the sender thread starts, the thread will not accept any more request. 
Sockets are owned solely by sender thread and therefore do not need any synchronization.  
Removing this restriction can be achieved by adding synchronization between sender thread and connection management thread, but the overhead was not trivial.  
For these reasons we decided to take more 'uncomfortable' way when implementing the server.

###### 4. (Optional) Run Validation
If you haven't compiled the validator yet, first run
>  mkdir JavaValidator/com  
>  scripts/compile_validator.sh

And if you don't have the validation file, run
>  cd scripts
>  ./create_validat.sh *cardata-path* *tolldata-path* *#lanes*

Then, you should locate the stream.out file from the Flink directory.  
stream.out is the output file from Linear Road Benchmark program. Then,
>  cd scripts
>  ./validate_output.sh *stream.out-path*
