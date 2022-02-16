#include <stdlib.h>
#include <stdio.h>

// multiply matrix
__global__ void multKernel(int* c, const int* a, const int* b) {
	// 2 dimensional index
	int x = threadIdx.x;
	int y = threadIdx.y;

	// Cij = Aik + Bkj
	for (int k = 0; k < (blockDim.x); k++) {
		c[x*5 + y] += (a[x*5 + k] * b[k*5 + y]);
	}
}

// print matrix
void printSquareMatrix(const int *matrix, const int WIDTH) {
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d\t", matrix[i*WIDTH + j]);
		}
		printf("\n");
	}
}

int main() {
	// host array
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };
	for (int y = 0; y < WIDTH; y++) {
		for (int x = 0; x < WIDTH; x++) {
			a[y][x] = y + x;
			b[y][x] = y + x;
		}
	}

	// device array
	int* dev_a, * dev_b, * dev_c = 0;
	cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

	// copy matrix from host to device
	cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

	// multiply Matrix
	dim3 DimBlock(WIDTH, WIDTH);
	multKernel << <1, DimBlock >> > (dev_c, dev_a, dev_b);

	// copy result matrix from device to host
	cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

	// free memory space
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	// print result matrix
	printf("행렬 A:\n");
	printSquareMatrix(*a, WIDTH);
	printf("\n행렬 B:\n");
	printSquareMatrix(*b, WIDTH);
	printf("\nA와 B의 행렬곱 결과행렬 C:\n");
	printSquareMatrix(*c, WIDTH);

	return 0;
}
