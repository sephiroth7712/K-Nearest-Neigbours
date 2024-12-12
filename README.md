
# Parallel K-Nearest Neighbors Implementation

A comprehensive implementation of the K-Nearest Neighbors (KNN) classification algorithm using multiple parallel computing paradigms. This project demonstrates advanced parallel and distributed programming techniques by implementing KNN using CUDA (GPU), Apache Hadoop, Apache Spark, MPI (distributed computing), OpenMP, and PThreads (shared memory parallelism).

## Overview

The K-Nearest Neighbors algorithm is implemented in seven different versions to showcase various parallel and distributed computing approaches:

1. Serial Implementation (baseline)
2. CUDA Implementation (single and multi-GPU)
3. Hadoop MapReduce Implementation
4. Apache Spark Implementation
5. MPI Implementation (distributed computing)
6. OpenMP Implementation (shared memory parallelism)
7. PThreads Implementation (explicit thread management)

Each implementation is optimized for its respective computing paradigm while maintaining classification accuracy.

## Technical Features

### Hadoop MapReduce Implementation
- Custom Writable types for efficient data serialization
- MapReduce workflow optimized for KNN computation
- Mapper phase for distance calculations
- Reducer phase for k-minimum element finding and classification
- Custom priority queue implementation for maintaining k-nearest neighbors
- Efficient partitioning strategy for balanced data distribution

### Spark Implementation
- Written in Scala using Spark's DataFrame API
- Utilizes Spark's built-in ML features and optimizations
- Efficient data partitioning and repartitioning strategies
- Leverages Spark's in-memory processing capabilities
- Custom UDF for distance calculations and neighbor finding
- Optimized for large-scale distributed processing

### CUDA Implementation
- Utilizes shared memory optimization for faster data access
- Implements efficient parallel distance calculation using 2D thread blocks
- Features both single-GPU and multi-GPU versions using MPI for inter-GPU communication
- Uses custom CUDA kernels for distance computation and k-minimum element finding
- Implements efficient parallel reduction techniques for finding k-nearest neighbors

### MPI Implementation
- Distributes workload across multiple processes
- Uses efficient data partitioning strategies for balanced computation
- Implements gather/scatter operations for result collection
- Features hybrid MPI+OpenMP version for hierarchical parallelism

### OpenMP Implementation
- Utilizes parallel for loops for workload distribution
- Implements thread-safe distance calculation and neighbor finding
- Features dynamic scheduling for load balancing
- Optimizes memory access patterns for cache efficiency

### PThreads Implementation
- Implements custom thread pool for work distribution
- Features efficient work queue management
- Uses thread-local storage for performance optimization
- Implements lock-free algorithms where possible

## Setup Instructions

### Prerequisites
```bash
# Required packages
- Apache Hadoop (>= 3.3.6)
- Apache Spark (>= 3.5.0)
- Scala (>= 2.12.17)
- CUDA Toolkit (>= 11.4)
- OpenMPI
- GCC Compiler (with OpenMP support)
- Maven
- Make build system
```

### Compilation

#### Hadoop Version
```bash
cd A3-Hadoop
mvn clean package
```

#### Spark Version
```bash
cd A3-Spark
mvn clean package
```

#### Other Versions
```bash
# CUDA versions
nvcc -o single_gpu GPU/single_gpu.cu
nvcc -o multi_gpu GPU/multi_gpu.cu

# MPI versions
mpic++ -o mpi MPI/mpi.cpp
mpic++ -o mpi_openmp MPI/mpi_openmp.cpp -fopenmp

# OpenMP version
g++ -o openmp Multi-threaded/openmp.cpp -fopenmp

# PThreads version
g++ -o threaded Multi-threaded/threaded.cpp -pthread

# Serial version
g++ -o serial serial.cpp
```

## Usage Examples

### Running Hadoop Version
```bash
hadoop jar target/mapreduce-1.0.jar mapreduce.KNN input/train.arff input/test.arff k output
```

### Running Spark Version
```bash
spark-submit --class spark.knn.KNearestNeighbours target/sparkml-1.0.jar
```

### Running Single GPU Version
```bash
./single_gpu datasets/train.arff datasets/test.arff k
```

### Running Multi-GPU Version
```bash
mpirun -np num_gpus ./multi_gpu datasets/train.arff datasets/test.arff k num_nodes
```

### Running MPI Version
```bash
mpirun -np num_processes ./mpi datasets/train.arff datasets/test.arff k
```

### Running OpenMP Version
```bash
./openmp datasets/train.arff datasets/test.arff k num_threads
```

### Running PThreads Version
```bash
./threaded datasets/train.arff datasets/test.arff k num_threads
```

Where:
- `datasets/train.arff`: Training dataset in ARFF format
- `datasets/test.arff`: Test dataset in ARFF format
- `k`: Number of nearest neighbors to consider
- `num_processes`: Number of MPI processes
- `num_threads`: Number of threads for OpenMP/PThreads
- `num_gpus`: Number of GPUs for multi-GPU version
- `num_nodes`: Number of nodes for distributed computation

## Performance Analysis

Each implementation outputs performance metrics including:
- Total execution time
- For GPU versions: Kernel execution time and memory transfer time
- For Hadoop/Spark versions: Job completion time and resource utilization
- Classification accuracy
- Number of instances processed
- Resource utilization (threads/processes/GPUs/containers)

Example output:
```
Single GPU: The 3-NN classifier for 3436 test instances and 61606 train instances required 18.875872 ms total time. Kernel and memcpy required 8.012352 ms. Kernel only required 7.421696 ms. Accuracy was 99.48%
```
## Runtimes (ms)
Runtimes are shown below for various implementations with varying number of threads/processes/GPUs/Nodes. 
The dataset contains 3436 test instances and 61606 train instances. All runtimes are calculated for a 3-NN classifier.
| # Threads/Processes/GPUs/Nodes | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Serial | 14056.00 | -- | -- | -- | -- | -- | -- | -- | -- |
| Threaded | 14021.00 | 7035.00 | 3533.00 | 1775.00 | 902.00 | 468.00 | 254.00 | 165.00 | 140.00 |
| OpenMP | 14472.00 | 7261.00 | 3655.00 | 1837.00 | 925.00 | 479.00 | 263.00 | 165.00 | 154.00 |
| MPI + OpenMP 1 node | 1868.00 | 509.67 | 264.33 | 138.33 | 143.33 | 159.00 | 170.00 | 161.67 | 164.33 |
| MPI + OpenMP 4 nodes | 900.67 | 141.33 | 97.67 | 117.67 | 99.00 | 85.00 | 93.33 | 126.67 | 168.67 |
| GPU | 7.78 | 106.54 | -- | -- | -- | -- | -- | -- | -- |
| Hadoop | 220533.00 | -- | -- | -- | -- | -- | -- | -- | -- |
| Spark | 10638.00 | -- | -- | -- | -- | -- | -- | -- | -- |


## Input Data Format

The implementation accepts ARFF (Attribute-Relation File Format) files containing:
- Numerical attributes for feature vectors
- Class labels as the last attribute
- Header information describing attributes
- Data instances in comma-separated format

## Project Structure
```
.
├── A3-Hadoop/
│   ├── pom.xml
│   └── src/main/java/mapreduce/
│       ├── CDInstanceWritable.java
│       ├── CDWritable.java
│       ├── KNN.java
│       ├── KNNMapper.java
│       ├── KNNReducer.java
│       └── KSmallestListPair.java
├── A3-Spark/
│   ├── pom.xml
│   └── src/main/scala/spark/knn/
│       └── KNearestNeighbours.scala
├── GPU/
│   ├── multi_gpu.cu
│   └── single_gpu.cu
├── MPI/
│   ├── mpi_openmp.cpp
│   └── mpi.cpp
├── Multi-threaded/
│   ├── openmp.cpp
│   └── threaded.cpp
└── serial.cpp
```

This project demonstrates proficiency in multiple parallel and distributed computing frameworks, showcasing implementations ranging from low-level thread management to high-level distributed computing frameworks like Hadoop and Spark. Each implementation is optimized for its respective paradigm while maintaining the core KNN algorithm's accuracy.
