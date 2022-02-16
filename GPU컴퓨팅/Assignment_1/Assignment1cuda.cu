#include <stdlib.h>
#include <stdio.h>
#define SIZE 10
 
__global__ void test(int *a, int *b){
    int i = threadIdx.x;
    b[i] = a[i] + 1;
}
 
int main(){
    int *a, *b;
    int *d_a, *d_b; 
 
    a = (int *)malloc(SIZE*sizeof(int));
    b = (int *)malloc(SIZE*sizeof(int));

    // 아래 assignment 에 해당하는 코드를 작성하여 전체 코드를 완성하고, CPP 파일과 동일한 output 이 출력되도록 하시오.

    /*Assignment 1-1: Allocate memory space for d_a and d_b using cudaMalloc function*/
    cudaMalloc((void**)&d_a, SIZE * sizeof(int));
    cudaMalloc((void**)&d_b, SIZE * sizeof(int));
    
    for (int i = 0; i<SIZE; ++i)
    {
        a[i] = i;
        b[i] = 0;
    }
    
    /*  Assignment 1-2: 
    Copy the contents of array 'a' in the host (CPU) memory 
    to the device (GPU) memory (d_a) using cudaMemcpy function
    */
    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    /*  Assignment 1-2:
    Copy the contents of array 'b' in the host (CPU) memory
    to the device (GPU) memory (d_b) using cudaMemcpy function
    */
    cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    test <<< 1, SIZE >>>(d_a, d_b); // launch test function
    /*  Assignment 1-3:
    Copy the results in array d_b in the device (GPU) memory
    to the host (CPU) memory using cudaMemcpy function
    */
    cudaMemcpy(b, d_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<SIZE; i++)
        printf("b[%d] = %d\n", i, b[i]);    // print the results
 
    free(a);    // free the host memory spaces
    free(b);    // free the host memory spaces
    
    cudaFree(d_a);    // free the device memory spaces 
    cudaFree(d_b);    // free the device memory spaces 
    return 0;
}