#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// fill array with random number
void fillArray(int* arr, int product, const int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        arr[i] = (rand() % 8 + 1) * product;   // 1~9 * product
    }
}

// print array
void printArray(const int* arr, const int SIZE) {
    printf("{%d", arr[0]);
    for (int i = 1; i < SIZE; i++) {
        printf(",%d", arr[i]);
    }
    printf("}");
}

__global__ void addKernel(int *d, const int *a, const int *b, const int *c){
    int i = threadIdx.x;
    d[i] = a[i] + b[i] + c[i];
}
 
int main(){
    const int SIZE = 5;
    
    // initialize host vectors
    srand(time(NULL));
    int a[SIZE] = { 0 };
    int b[SIZE] = { 0 };
    int c[SIZE] = { 0 };
    int d[SIZE] = { 0 };
    fillArray(a, 100, SIZE);
    fillArray(b, 10, SIZE);
    fillArray(c, 1, SIZE);

    // device vectors
    int* dev_b = 0;
    int* dev_a = 0;
    int* dev_c = 0;
    int* dev_d = 0;

    // memory allocation
    cudaMalloc((void**)&dev_d, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    
    // copy memory from host to device
    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // vertor addition
    addKernel<<<1, SIZE >>>(dev_d, dev_a, dev_b, dev_c);

    // copy result from device to host
    cudaMemcpy(d, dev_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    printArray(a, SIZE);
    printf(" + ");
    printArray(b, SIZE);
    printf(" + ");
    printArray(c, SIZE);
    printf(" = ");
    printArray(d, SIZE);
    printf("\n");

    // free the device memory spaces
    cudaFree(dev_d);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}