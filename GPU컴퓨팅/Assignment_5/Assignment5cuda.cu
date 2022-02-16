#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <chrono>

#define GRIDSIZE (16*1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(float* ptr, unsigned int size)
{
	while (size--)
	{
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

void adjDiff_host(float* dst, const float* src, unsigned int size)
{
	for (int i = 1; i < size; ++i)
	{
		dst[i] = src[i] - src[i - 1];
	}
}

__global__ void adjDiff_global(float* result, float* input)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0)
	{
		float x_i = input[i];	// global >>> reg
		float x_i_m1 = input[i - 1];	// global >>> reg
		result[i] = x_i - x_i_m1;	// calc, store to global
	}
}

__global__ void adjDiff_shared(float* result, float* input)
{
	// shared mem
	__shared__ float s_data[BLOCKSIZE];
	unsigned int tx = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[tx] = input[i];
	// barrier	
	__syncthreads();

	// in thread block
	if (tx > 0)
	{
		// calc and store result to global mem
		result[i] = s_data[tx] - s_data[tx - 1];
	}
	// tx = 0, i > 0
	// use out value of shared mem
	else if (i > 0)
	{
		// use global input instead of shared mem
		result[i] = s_data[tx] - input[i - 1];
	}
}

int main(void)
{
	// host variables
	float* pSource = NULL;
	float* pResult = NULL;
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));

	// generate input source data
	genData(pSource, TOTALSIZE);
	
	// CUDA variables
	float* pSourceDev = NULL;
	float* pResultDev = NULL;
	cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float));
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);

	// elapse time variables
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
	std::chrono::nanoseconds duration_nano;
	std::chrono::system_clock::time_point func_start;
	std::chrono::system_clock::time_point func_end;
	std::chrono::nanoseconds func_duration_nano;


	//* calc adjacent difference in host *//
	// start
	start = std::chrono::system_clock::now();
	// calc in host
	pResult[0] = 0.0F; // exceptional case i = 0
	adjDiff_host(pResult, pSource, TOTALSIZE);
	// end
	end = std::chrono::system_clock::now();
	duration_nano = end - start;
	// print result
	printf("Elapsed Time (adjDiff_host):\t\t\t%lld\n", duration_nano);


	//* calc adjacent difference in CUDA global *//
	// start
	start = std::chrono::system_clock::now();
	pResult[0] = 0.0f;	// exceptional case for i = 0;
	// cuda mem cpy from h to d
	cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);
	
	// func start
	func_start = std::chrono::system_clock::now();
	//cuda launch the kernel adjdiff in CUDA Global mem
	adjDiff_global <<<dimGrid, dimBlock>>> (pResultDev, pSourceDev);
	// func end
	func_end = std::chrono::system_clock::now();
	
	// cuda memcpy from d to h
	cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
	// end
	end = std::chrono::system_clock::now();
	
	// print result
	func_duration_nano = func_end - func_start;
	duration_nano = end - start;
	printf("Elapsed Time (adjDiff_global only):\t\t%8lld\n", func_duration_nano);
	printf("Elapsed Time (adjDiff_global with memcpy):\t%lld\n", duration_nano);


	//* calc adjacent difference in CUDA shared *//
	// start
	start = std::chrono::system_clock::now();
	pResult[0] = 0.0F;	// exceptional case for i = 0;
	// CUDA mem cpy from h to d
	cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);

	// func start
	func_start = std::chrono::system_clock::now();
	// CUDA launch the kernel adjDiff_shared
	adjDiff_shared <<<dimGrid, dimBlock>>> (pResultDev, pSourceDev);
	// func end
	func_end = std::chrono::system_clock::now();
	
	// CUDA memcpy from d to h
	cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
	// end
	end = std::chrono::system_clock::now();
	
	// print result
	func_duration_nano = func_end - func_start;
	duration_nano = end - start;
	printf("Elapsed Time (adjDiff_shared only):\t\t%8lld\n", func_duration_nano);
	printf("Elapsed Time (adjDiff_shared with memcpy):\t%lld\n", duration_nano);


	// free mem
	free(pSource);
	free(pResult);
	cudaFree(pSourceDev);
	cudaFree(pResultDev);
}