#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <pthread.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

struct thread_data
{
    float *test_data;
    int test_num_instances;
    float *train_data;
    int *predictions_ptr;
};

int k, train_num_instances, num_attributes, num_classes;
;
unsigned char killthreads = 0;

// Calculates the distance between two instances
float distance(float *instance_A, float *instance_B, int num_attributes)
{
    float sum = 0;

    for (int i = 0; i < num_attributes - 1; i++)
    {
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

void *KNN_thread(void *ptr)
{

    struct thread_data data = *(struct thread_data *)ptr;

    float *test_matrix = data.test_data;
    float *train_matrix = data.train_data;
    int test_num_instances = data.test_num_instances;
    int *predictions = data.predictions_ptr;

    float *candidates = (float *)calloc(k * 2, sizeof(float));
    for (int i = 0; i < 2 * k; i++)
    {
        candidates[i] = FLT_MAX;
    }

    int *classCounts = (int *)calloc(num_classes, sizeof(int));
    for (int queryIndex = 0; queryIndex < test_num_instances; queryIndex++)
    {
        for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++)
        {

            float dist = distance(&test_matrix[queryIndex * num_attributes], &train_matrix[keyIndex * num_attributes], num_attributes);

            // Add to our candidates
            for (int c = 0; c < k; c++)
            {
                if (dist < candidates[2 * c])
                {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--)
                    {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }

                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = train_matrix[keyIndex * num_attributes + num_attributes - 1]; // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++)
        {
            classCounts[(int)candidates[2 * i + 1]] += 1;
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

        // Reset arrays
        for (int i = 0; i < 2 * k; i++)
        {
            candidates[i] = FLT_MAX;
        }
        memset(classCounts, 0, num_classes * sizeof(int));
    }

    free(candidates);
    free(classCounts);

    pthread_exit(0);
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int *KNN(ArffData *train, ArffData *test, int k, int num_threads)
{

    ::k = k;
    struct thread_data thread_data_array[num_threads];
    int *predictions = (int *)calloc(test->num_instances(), sizeof(int));

    /*************************************************************
    *** Complete this code and return the array of predictions ***
    **************************************************************/
    pthread_t *threads;

    num_classes = train->num_classes();
    num_attributes = train->num_attributes();
    train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    int test_num_instances_per_thread = test_num_instances / num_threads;
    int remainder = test_num_instances - test_num_instances_per_thread * num_threads;

    threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    float *train_matrix = train->get_dataset_matrix();
    float *test_matrix = test->get_dataset_matrix();

    int last_test_index = 0;
    for (int i = 0; i < num_threads; i++)
    {
        thread_data_array[i].train_data = &train_matrix[0];
        thread_data_array[i].test_data = &test_matrix[last_test_index * num_attributes];
        thread_data_array[i].predictions_ptr = &predictions[last_test_index];

        if (remainder > 0)
        {
            thread_data_array[i].test_num_instances = test_num_instances_per_thread + 1;
            last_test_index += test_num_instances_per_thread + 1;
            remainder--;
        }
        else
        {
            thread_data_array[i].test_num_instances = test_num_instances_per_thread;
            last_test_index += test_num_instances_per_thread;
        }

        pthread_create(&threads[i], NULL, KNN_thread, (void *)&thread_data_array[i]);
    }

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    free(threads);
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
    if (argc != 5)
    {
        printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);
    int num_threads = strtol(argv[4], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    struct timespec start, end;
    int *predictions = NULL;

    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    predictions = KNN(train, test, k, num_threads);

    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int *confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for threaded with %d threads. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int)time_difference, accuracy, num_threads);

    free(predictions);
    free(confusionMatrix);
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/