#include<stdio.h>
#include<fstream>

__global__ void Conv3x3_2d(double* input, double* weight, double* output)
{
    // shared mem
    __shared__ double s_input[1024];
    __shared__ double s_weight[9];
    
    // index
    unsigned int tx = threadIdx.x;  // 0~31
    unsigned int ty = threadIdx.y;  // 0~31
    unsigned int bx = blockIdx.x;   // 0~6
    unsigned int by = blockIdx.y;   // 0~6
    unsigned int bz = blockIdx.z;   // 0~299
    
    // get shared data
    s_input[ty*32 + tx] = input[bz*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)];
    if(ty == 0 && tx < 9) s_weight[tx] = weight[tx];

    // barrier
    __syncthreads();

    // calculate
    double sum = 0.0;
    if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
    {
        sum += s_input[ (ty-1)*32 + tx-1] * s_weight[0];
        sum += s_input[ (ty-1)*32 + tx  ] * s_weight[1];
        sum += s_input[ (ty-1)*32 + tx+1] * s_weight[2];

        sum += s_input[ ty*32     + tx-1] * s_weight[3];
        sum += s_input[ ty*32     + tx  ] * s_weight[4];
        sum += s_input[ ty*32     + tx+1] * s_weight[5];
    
        sum += s_input[ (ty+1)*32 + tx-1] * s_weight[6];
        sum += s_input[ (ty+1)*32 + tx  ] * s_weight[7];
        sum += s_input[ (ty+1)*32 + tx+1] * s_weight[8];
    }
    else
    {
        sum += s_weight[0] * input[ bz*226*226 + (by*32 + ty    )*226 + (bx*32 + tx)];
        sum += s_weight[1] * input[ bz*226*226 + (by*32 + ty    )*226 + (bx*32 + tx + 1)];
        sum += s_weight[2] * input[ bz*226*226 + (by*32 + ty    )*226 + (bx*32 + tx + 2)];

        sum += s_weight[3] * input[ bz*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx)];
        sum += s_weight[4] * s_input[ty*32 + tx];
        sum += s_weight[5] * input[ bz*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 2)];
    
        sum += s_weight[6] * input[ bz*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx)];
        sum += s_weight[7] * input[ bz*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx + 1)];
        sum += s_weight[8] * input[ bz*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx + 2)];
    }
    
    output[bz*224*224 + (by*32 + ty)*224 + (bx*32 + tx)] = sum;
}

__global__ void Conv1x1_3d(double* input, double* weight, double* output)
{
    // shared mem
    __shared__ double s_weight[300];

    // index
    unsigned int tx = threadIdx.x;  // 0~31
    unsigned int ty = threadIdx.y;  // 0~31
    unsigned int bx = blockIdx.x;   // 0~6
    unsigned int by = blockIdx.y;   // 0~6
    unsigned int bz = blockIdx.z;   // 0~899
    
    // get shared data
    if(ty*32 + tx < 300)    s_weight[ty*32 + tx] = weight[bz*300 + (ty*32 + tx)];

    // barrier
    __syncthreads();

    // calculate
    double sum = 0.0;
    for(int i = 0; i < 300; i=i+6)
    {
        sum += input[(i  )*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i];
        sum += input[(i+1)*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i+1];
        sum += input[(i+2)*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i+2];
        sum += input[(i+3)*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i+3];
        sum += input[(i+4)*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i+4];
        sum += input[(i+5)*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)] * s_weight[i+5];
    }

    output[(bz*224*224) + (by*32 + ty)*224 + (bx*32 + tx)] = sum;
}

__global__ void Conv3x3_3d(double* input, double* weight, double* output)
{
    // shared mem
    __shared__ double s_input[1024];
    __shared__ double s_weight[900];

    // index
    unsigned int tx = threadIdx.x;  // 0~31
    unsigned int ty = threadIdx.y;  // 0~31
    unsigned int bx = blockIdx.x;   // 0~6
    unsigned int by = blockIdx.y;   // 0~6
    unsigned int bz = blockIdx.z;   // 0~899
    
    // calculate
    double sum = 0.0;
    for(int i = 0; i < 300; i++)
    {
        // get shared data
        s_input[ty*32 + tx] = input[i*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 1)];
        //if(ty == 0 && tx < 9) s_weight[tx] = weight[bz*300*9 + i*9 + tx];
        if(i%100 == 0 && ty*32+tx < 900)
            s_weight[ty*32 + tx] = weight[bz*300*9 + ty*32 + tx];

        // barrier
        __syncthreads();

        if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
        {
            sum += s_input[ (ty-1)*32 + tx-1] * s_weight[0];
            sum += s_input[ (ty-1)*32 + tx  ] * s_weight[1];
            sum += s_input[ (ty-1)*32 + tx+1] * s_weight[2];
    
            sum += s_input[ ty*32     + tx-1] * s_weight[3];
            sum += s_input[ ty*32     + tx  ] * s_weight[4];
            sum += s_input[ ty*32     + tx+1] * s_weight[5];
        
            sum += s_input[ (ty+1)*32 + tx-1] * s_weight[6];
            sum += s_input[ (ty+1)*32 + tx  ] * s_weight[7];
            sum += s_input[ (ty+1)*32 + tx+1] * s_weight[8];
        }
        else
        {
            sum += s_weight[0] * input[ i*226*226 + (by*32 + ty    )*226 + (bx*32 + tx)];
            sum += s_weight[1] * input[ i*226*226 + (by*32 + ty    )*226 + (bx*32 + tx + 1)];
            sum += s_weight[2] * input[ i*226*226 + (by*32 + ty    )*226 + (bx*32 + tx + 2)];
    
            sum += s_weight[3] * input[ i*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx)];
            sum += s_weight[4] * s_input[ty*32 + tx];
            sum += s_weight[5] * input[ i*226*226 + (by*32 + ty + 1)*226 + (bx*32 + tx + 2)];
        
            sum += s_weight[6] * input[ i*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx)];
            sum += s_weight[7] * input[ i*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx + 1)];
            sum += s_weight[8] * input[ i*226*226 + (by*32 + ty + 2)*226 + (bx*32 + tx + 2)];
        }

        // barrier
        __syncthreads();    
    }

    output[(bz*224*224) + (by*32 + ty)*224 + (bx*32 + tx)] = sum;
}

void input(double * input)
{
    for(int z=0;z<300;z++)
        for(int i=0;i<226;i++)
            for(int j=0;j<226;j++)
            {
                if(i == 0 || j == 0 || j == 225 || i == 225)
                    input[z*226*226+i*226+j] = 0;
                else
                    input[z*226*226+i*226+j] = 1;
            }
}

void Init_weight3x3_3d(double * weight)
{
    for(int k=0;k<900;k++)
        for(int z=0;z<300;z++)
            for(int i=0;i<3;i++)
                for(int j=0;j<3;j++){
                    if((i*3 + j) % 2 == 0)
                        weight[k*3*3*300 + z*3*3+i*3+j] = 1;
                    else
                        weight[k*3*3*300 + z*3*3+i*3+j] = -1;
                }
}

void Init_weight3x3_2d(double * weight)
{
    for(int i=0;i<9;i++)
        weight[i] = 1;
}

void Init_weight1x1_3d(double * weight)
{
    for(int k = 0;k<900;k++)
        for(int z=0;z<300;z++){
            if(k % 2 == 0)
                weight[300*k + z] = 1;
            else
                weight[300*k + z] = -1;
        }
}

void save_file(double * output3x3_3d,double * output1x1_3d,double * output3x3_2d)
{
    FILE * fp = fopen("3d_Result3x3.txt","w");
    FILE * fp2 = fopen("3d_Result1x1.txt","w");
    FILE * fp3 = fopen("2d_Result3x3.txt","w");
    for(int i=0;i<224*224*900;i++)
    {
        fprintf(fp,"%f\n",output3x3_3d[i]);
        fprintf(fp2,"%f\n",output1x1_3d[i]);
    }
    for(int i=0;i<224*224*300;i++)
        fprintf(fp3,"%f\n",output3x3_2d[i]);

    fclose(fp3);
    fclose(fp2);
    fclose(fp);
}
int main() {

    double* input_3d = (double*)malloc(sizeof(double) * 226 * 226 * 300);    //input Feature Map

    double* output3x3_2d = (double*)malloc(sizeof(double) * 224 * 224 * 300);//Output Feature Map
    double* output1x1_3d = (double*)malloc(sizeof(double) * 224 * 224 * 900);
    double* output3x3_3d = (double*)malloc(sizeof(double) * 224 * 224 * 900);

    double* weight3x3_2d = (double*)malloc(sizeof(double) * 3 * 3);		//Weight
    double* weight1x1_3d = (double*)malloc(sizeof(double) * 1 * 1 * 300 * 900);
    double* weight3x3_3d = (double*)malloc(sizeof(double) * 3 * 3 * 300 * 900);

    double* g_input, * g_output3x3_3d, * g_output1x1_3d, * g_output3x3_2d, * g_weight3x3_3d, * g_weight1x1_3d, * g_weight3x3_2d;

    cudaEvent_t start, stop3x3_3d, stop1x1_3d, stop3x3_2d;

    cudaEventCreate(&start);
    cudaEventCreate(&stop3x3_3d);
    cudaEventCreate(&stop1x1_3d);
    cudaEventCreate(&stop3x3_2d);
    //Initialization Input Feature Map & Weight
    input(input_3d);
    Init_weight3x3_3d(weight3x3_3d);
    Init_weight1x1_3d(weight1x1_3d);
    Init_weight3x3_2d(weight3x3_2d);

    cudaMalloc((void**)&g_input, sizeof(double) * 226 * 226 * 300);
    cudaMalloc((void**)&g_output3x3_3d, sizeof(double) * 224 * 224 * 900);
    cudaMalloc((void**)&g_output1x1_3d, sizeof(double) * 224 * 224 * 900);
    cudaMalloc((void**)&g_output3x3_2d, sizeof(double) * 224 * 224 * 300);
    cudaMalloc((void**)&g_weight3x3_3d, sizeof(double) * 3 * 3 * 300 * 900);
    cudaMalloc((void**)&g_weight1x1_3d, sizeof(double) * 1 * 1 * 300 * 900);
    cudaMalloc((void**)&g_weight3x3_2d, sizeof(double) * 3 * 3);

    cudaMemcpy(g_input, input_3d, sizeof(double) * 226 * 226 * 300, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight3x3_3d, weight3x3_3d, sizeof(double) * 3 * 3 * 300 * 900, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight1x1_3d, weight1x1_3d, sizeof(double) * 1 * 1 * 300 * 900, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight3x3_2d, weight3x3_2d, sizeof(double) * 3 * 3, cudaMemcpyHostToDevice);

    /*
    Project
    Block 및 Grid 선언 자유, 주어진 3개의 Kernel Conv3x3_3d, Conv1x1_3d, Conv3x3_2d를 구현(Kernel명 및 Argument 유지)
    가능한 빠른 Performance를 가지는 Kernel을 구현할 것

    결과는 Text File을 통해서 확인, cudaEvent 관련 코드는 성능 측정을 위한 코드이니 수정하지 말것

    Kernel 별 배점
    Conv3x3_2d = 20%
    Conv1x1_3d = 35%
    Conv3x3_3d = 45%
    */
    ////////////////////////////////////////아래의 3개 Kernel을 구현 ///////////////////////////////////
	dim3 dimGrid_2d(7, 7, 300);
	dim3 dimGrid_3d(7, 7, 900);
	dim3 dimBlock(32, 32, 1);

    cudaEventRecord(start);
    Conv3x3_2d <<<dimGrid_2d, dimBlock>>> (g_input, g_weight3x3_2d, g_output3x3_2d);
    cudaEventRecord(stop3x3_2d);

    Conv1x1_3d <<<dimGrid_3d, dimBlock>>> (g_input, g_weight1x1_3d, g_output1x1_3d);
    cudaEventRecord(stop1x1_3d);

    Conv3x3_3d <<<dimGrid_3d, dimBlock>>> (g_input, g_weight3x3_3d, g_output3x3_3d);
    cudaEventRecord(stop3x3_3d);

    cudaEventSynchronize(stop3x3_3d);    
/////////////////////////////////////////////////////////////////////////////////////////////////

    float milliseconds[3]={0};
    cudaEventElapsedTime(&milliseconds[0],start,stop3x3_2d);
    cudaEventElapsedTime(&milliseconds[1],stop3x3_2d,stop1x1_3d);
    cudaEventElapsedTime(&milliseconds[2],stop1x1_3d,stop3x3_3d);
    printf("Execution Time \n Convolution3x3_2d : %f\n Convolution1x1_3d : %f\n Convolution3x3_3d : %f\n",milliseconds[0],milliseconds[1],milliseconds[2]);
    
    cudaMemcpy(output3x3_3d,g_output3x3_3d,sizeof(double)*224*224*900,cudaMemcpyDeviceToHost);
    cudaMemcpy(output1x1_3d,g_output1x1_3d,sizeof(double)*224*224*900,cudaMemcpyDeviceToHost);
    cudaMemcpy(output3x3_2d,g_output3x3_2d,sizeof(double)*224*224*300,cudaMemcpyDeviceToHost);

    save_file(output3x3_3d,output1x1_3d,output3x3_2d);
    cudaFree(g_input);
    cudaFree(g_weight3x3_3d);
    cudaFree(g_weight3x3_2d);
    cudaFree(g_weight1x1_3d);
    cudaFree(g_output3x3_3d);
    cudaFree(g_output3x3_2d);
    cudaFree(g_output1x1_3d);

    free(output3x3_3d);
    free(output1x1_3d);
    free(output3x3_2d);
    free(input_3d);
    free(weight3x3_3d);
    free(weight1x1_3d);
    free(weight3x3_2d);
}
