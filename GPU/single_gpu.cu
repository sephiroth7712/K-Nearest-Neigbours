#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
// Use "module load mpi devtoolset/9 Cuda11.4" on huff
// Run test on huff as "salloc --partition GPU --qos gpu --nodes 1 --ntasks-per-node 1 --cpus-per-task 1 mpirun -np 1 ./single_gpu datasets/large-train.arff datasets/large-test.arff 3"

float kernel_runtime;
float kernel_memcpy_runtime;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/***************************************************/
/*** WRITE THE CODE OF THE KERNEL FUNCTIONS HERE ***/
/***************************************************/

// Calculate the distance matrix for every point in the test matrix to every point in the train matrix
__global__ void distance(float *train, float *test, float *distance, int train_num_instances, int test_num_instances, int num_attributes)
{
    extern __shared__ float sdata[];
    float *train_shared = &sdata[0];
    float *test_shared = &sdata[blockDim.x * num_attributes];
    int blockThreadId = blockDim.x * threadIdx.y + threadIdx.x;
    int train_attribute = blockIdx.x * blockDim.x * num_attributes + blockDim.x * threadIdx.y + threadIdx.x;
    int test_attribute = blockIdx.y * blockDim.y * num_attributes + (blockDim.x * threadIdx.y + threadIdx.x) - blockDim.x * num_attributes;

    // Load the data from the train matrix into shared memory
    if ((blockThreadId < blockDim.x * num_attributes) && (train_attribute < num_attributes * train_num_instances))
    {
        train_shared[blockThreadId] = train[train_attribute];
    }
    // Load the data from the test matrix into shared memory
    else if ((blockThreadId >= blockDim.x * num_attributes) && (blockThreadId < 2 * blockDim.x * num_attributes) && (test_attribute < num_attributes * test_num_instances))
    {
        test_shared[blockThreadId - blockDim.x * num_attributes] = test[test_attribute];
    }
    __syncthreads();

    int train_instance = blockIdx.x * blockDim.x + threadIdx.x;
    int test_instance = blockIdx.y * blockDim.y + threadIdx.y;
    if ((train_instance < train_num_instances) && (test_instance < test_num_instances))
    {
        float sum = 0.0f;
        for (int i = 0; i < num_attributes - 1; i++)
        {
            float temp = train_shared[(threadIdx.x * num_attributes) + i] - test_shared[(threadIdx.y * num_attributes) + i];
            sum += temp * temp;
        }
        distance[(test_instance * train_num_instances) + train_instance] = sqrt(sum);
    }
}

// Calculate the K minimum elements for every test point (i.e every row in the distance matrix)
__global__ void findKMin(float *distances, int *minimum_indexes, int train_num_instances, int k)
{
    int train_instance = blockIdx.x * blockDim.x + threadIdx.x;
    int test_instance = blockIdx.y;
    extern __shared__ float sdata[];
    int *heap_indexes = (int *)sdata;
    float *heap_distances = (float *)&heap_indexes[blockDim.x * k];
    float curr_distance;
    int curr_index;
    if (train_instance < train_num_instances)
    {
        // curr_distance = distances[test_instance * train_num_instances + train_instance];
        // curr_index = train_instance;

        // Initialise the heap with the first distances assigned to this thread
        for (int i = 0; i < k; i++)
        {
            heap_indexes[i * blockDim.x + train_instance] = -1;
            heap_distances[i * blockDim.x + train_instance] = FLT_MAX;
        }
    }
    __syncthreads();

    /*
        Iterate through the row of the distance matrix with a stride of blockDim.x (256)
        This will ensure that at the end of the loop we have a list of K minimum elements for every element seen by the thread
    */
    for (int i = train_instance; i < train_num_instances; i += blockDim.x)
    {
        curr_distance = distances[test_instance * train_num_instances + i];
        curr_index = i;
        for (int j = k - 1; j >= 0; j--)
        {
            /*
                Check if the current element is greater than the largest element in the heap of the thread
                If yes, then rearrange the heap to accomodate the current element
            */
            if (heap_distances[(j * blockDim.x) + train_instance] >= curr_distance)
            {
                if (j == k - 1)
                {
                    heap_distances[(j * blockDim.x) + train_instance] = curr_distance;
                    heap_indexes[(j * blockDim.x) + train_instance] = curr_index;
                }
                else
                {
                    for (int l = k - 1; l > j; l--)
                    {
                        heap_distances[(l * blockDim.x) + train_instance] = heap_distances[((l - 1) * blockDim.x) + train_instance];
                        heap_indexes[(l * blockDim.x) + train_instance] = heap_indexes[((l - 1) * blockDim.x) + train_instance];
                    }

                    heap_distances[(j * blockDim.x) + train_instance] = curr_distance;
                    heap_indexes[(j * blockDim.x) + train_instance] = curr_index;
                }
            }
        }
    }
    __syncthreads();

    /*
        Every 32nd thread will find the minimum K elements by checking the minimum K elements calculated by the subsequent 32 threads
        For the given blockDim.x size, this will give us 8 k-minimum values (256/32 = 8)
    */
    if (threadIdx.x % 16 == 0)
    {
        for (int i = threadIdx.x; i < threadIdx.x + 16; i++)
        {
            for (int j = k - 1; j >= 0; j--)
            {
                if (heap_distances[(j * blockDim.x) + threadIdx.x] >= heap_distances[(j * blockDim.x) + i])
                {
                    if (j == k - 1)
                    {
                        heap_distances[(j * blockDim.x) + threadIdx.x] = heap_distances[(j * blockDim.x) + i];
                        heap_indexes[(j * blockDim.x) + threadIdx.x] = heap_indexes[(j * blockDim.x) + i];
                    }
                    else
                    {
                        for (int l = k - 1; l > j; l--)
                        {
                            heap_distances[(l * blockDim.x) + threadIdx.x] = heap_distances[((l - 1) * blockDim.x) + threadIdx.x];
                            heap_indexes[(l * blockDim.x) + threadIdx.x] = heap_indexes[((l - 1) * blockDim.x) + threadIdx.x];
                        }

                        heap_distances[(j * blockDim.x) + threadIdx.x] = heap_distances[(j * blockDim.x) + i];
                        heap_indexes[(j * blockDim.x) + threadIdx.x] = heap_indexes[(j * blockDim.x) + i];
                    }
                }
            }
        }
    }
    __syncthreads();

    /*
        Have one thread find the global K-minimum values by scanning the 8 K-minimum values calculated above
    */
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < blockDim.x / 16; i++)
        {
            for (int j = k - 1; j >= 0; j--)
            {
                if (heap_distances[(j * blockDim.x) + threadIdx.x] >= heap_distances[(j * blockDim.x) + i * 16])
                {
                    if (j == k - 1)
                    {
                        heap_distances[(j * blockDim.x) + threadIdx.x] = heap_distances[(j * blockDim.x) + i * 16];
                        heap_indexes[(j * blockDim.x) + threadIdx.x] = heap_indexes[(j * blockDim.x) + i * 16];
                    }
                    else
                    {
                        for (int l = k - 1; l > j; l--)
                        {
                            heap_distances[(l * blockDim.x) + threadIdx.x] = heap_distances[((l - 1) * blockDim.x) + threadIdx.x];
                            heap_indexes[(l * blockDim.x) + threadIdx.x] = heap_indexes[((l - 1) * blockDim.x) + threadIdx.x];
                        }

                        heap_distances[(j * blockDim.x) + threadIdx.x] = heap_distances[(j * blockDim.x) + i * 16];
                        heap_indexes[(j * blockDim.x) + threadIdx.x] = heap_indexes[(j * blockDim.x) + i * 16];
                    }
                }
            }
        }
        for (int i = 0; i < k; i++)
        {
            minimum_indexes[test_instance * k + i] = heap_indexes[i * blockDim.x + threadIdx.x];
        }
    }
}

// Implements a kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int *KNN(ArffData *train, float *train_matrix, ArffData *test, float *test_matrix, int k)
{

    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *)calloc(test_num_instances, sizeof(int));
    int *minimum_indexes = (int *)calloc(test_num_instances * k, sizeof(int));
    int *classCounts = (int *)calloc(num_classes, sizeof(int));

    // Declare timers
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEventKernel, stopEventKernel;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&startEventKernel);
    cudaEventCreate(&stopEventKernel);

    // Memory allocation on the GPU
    /********************************************************************/
    /*** WRITE THE CODE OF THE MEMORY ALLOCATION ON THE GPU HERE HERE ***/
    /********************************************************************/
    float *d_train, *d_test, *d_distances;
    int *d_predictions, *d_minimum_indexes;

    cudaMalloc(&d_train, train_num_instances * num_attributes * sizeof(float));
    cudaMalloc(&d_test, test_num_instances * num_attributes * sizeof(float));
    cudaMalloc(&d_distances, test_num_instances * train_num_instances * sizeof(float));
    cudaMalloc(&d_predictions, test_num_instances * sizeof(int));
    cudaMalloc(&d_minimum_indexes, test_num_instances * k * sizeof(int));

    // Start timer measuring kernel + memory copy times
    cudaEventRecord(startEvent, 0);

    // Memory copy from host to device
    /**************************************************************/
    /*** WRITE THE CODE OF THE MEMORY COPY FROM CPU TO GPU HERE ***/
    /**************************************************************/
    cudaMemcpy(d_train, train_matrix, train_num_instances * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, test_matrix, test_num_instances * num_attributes * sizeof(float), cudaMemcpyHostToDevice);

    // Start timer measuring kernel time only
    cudaEventRecord(startEventKernel, 0);

    // Execute the kernels
    /***********************************************/
    /*** WRITE THE CODE OF THE KERNEL CALLS HERE ***/
    /***********************************************/
    int threadsPerBlockDim = 24;
    int gridDimSizeX = (train_num_instances + threadsPerBlockDim - 1) / threadsPerBlockDim;
    int gridDimSizeY = (test_num_instances + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
    dim3 gridSize(gridDimSizeX, gridDimSizeY);

    // Calculate the distance matrix
    distance<<<gridSize, blockSize, 2 * threadsPerBlockDim * num_attributes * sizeof(float)>>>(d_train, d_test, d_distances, train_num_instances, test_num_instances, num_attributes);

    threadsPerBlockDim = 128;
    dim3 blockSize2(threadsPerBlockDim);
    gridDimSizeX = 1;
    gridDimSizeY = test_num_instances;
    dim3 gridSize2(gridDimSizeX, gridDimSizeY);
    unsigned smem_size = threadsPerBlockDim * k * sizeof(float) + threadsPerBlockDim * k * sizeof(int);

    // Find the indices of the K-minimum values for each row of the distance matrix
    findKMin<<<gridSize2, blockSize2, smem_size>>>(d_distances, d_minimum_indexes, train_num_instances, k);

    // Wait for kernel completion
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(minimum_indexes, d_minimum_indexes, k * test_num_instances * sizeof(int), cudaMemcpyDeviceToHost);

    // Derive the class of the test element using its k-minimum distances
    for (int queryIndex = 0; queryIndex < test_num_instances; queryIndex++)
    {
        for (int j = 0; j < k; j++)
        {
            classCounts[(int)train_matrix[num_attributes * minimum_indexes[queryIndex * k + j] + num_attributes - 1]] += 1;
        }
        int max_value = -1;
        int max_class = 0;
        for (int i = 0; i < num_classes; i++)
        {
            if (classCounts[i] > max_value)
            {
                max_value = classCounts[i];
                max_class = i;
            }
        }

        // Make prediction with
        predictions[queryIndex] = max_class;

        memset(classCounts, 0, num_classes * sizeof(int));
    }

    // Stop timer measuring kernel time only
    cudaEventRecord(stopEventKernel, 0);

    // Stop timer measuring kernel + memory copy times
    cudaEventRecord(stopEvent, 0);

    // Calculate elapsed time
    cudaEventSynchronize(stopEvent);
    cudaEventSynchronize(stopEventKernel);
    cudaEventElapsedTime(&kernel_memcpy_runtime, startEvent, stopEvent);
    cudaEventElapsedTime(&kernel_runtime, startEventKernel, stopEventKernel);

    // Free memory
    /**********************************************/
    /*** WRITE THE CODE TO FREE GPU MEMORY HERE ***/
    /**********************************************/
    free(classCounts);
    free(minimum_indexes);
    cudaFree(d_train);
    cudaFree(d_test);
    cudaFree(d_distances);
    cudaFree(d_predictions);
    cudaFree(d_minimum_indexes);
    return predictions;
}

int *computeConfusionMatrix(int *predictions, ArffData *dataset)
{
    int *confusionMatrix = (int *)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for (int i = 0; i < dataset->num_instances(); i++)
    { // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int *confusionMatrix, ArffData *dataset)
{
    int successfulPredictions = 0;

    for (int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return 100 * successfulPredictions / (float)dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: ./program datasets/train.arff datasets/test.arff k\n");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    // Pointers representing the dataset as a 2D matrix size num_instances x num_attributes
    float *train_matrix = train->get_dataset_matrix();
    float *test_matrix = test->get_dataset_matrix();

    int *predictions = NULL;

    // Initialize time measurement
    float time_difference;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    predictions = KNN(train, train_matrix, test, test_matrix, k);

    // Stop time measurement
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_difference, startEvent, stopEvent);

    // Compute the confusion matrix
    int *confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    printf("Single GPU: The %i-NN classifier for %lu test instances and %lu train instances required %f ms total time. Kernel and memcpy required %f ms. Kernel only required %f ms. Accuracy was %.2f%\n", k, test->num_instances(), train->num_instances(), time_difference, kernel_memcpy_runtime, kernel_runtime, accuracy);

    free(train_matrix);
    free(test_matrix);
    free(predictions);
    free(confusionMatrix);
}