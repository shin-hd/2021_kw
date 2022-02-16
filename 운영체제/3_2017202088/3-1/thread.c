#include <stdio.h>
#include <sys/types.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <linux/unistd.h>

#define MAX_PROCESSES 64

FILE* f_read;

// read file and add two numbers
int read_and_add()
{
	int a = 0, b = 0;
	char line[8] = { 0, };

	// get numbers of file
	fgets(line, 8, f_read);
	a = atoi(line);
	fgets(line, 8, f_read);
	b = atoi(line);

	return a+b;
}

void* thread_func(void *arg)
{
	int *sum = 0, half = (int)arg;
	sum = malloc(sizeof(int));

	// at leaves, read file, add numbers and return sum
	if(half == 1) {
		*sum = read_and_add();
		return sum;
	}

	half /= 2;
	int *a = 0, *b = 0;
	pthread_t tid[2];

	// create thread calling thread_func(half)
	pthread_create(&tid[0], NULL, thread_func, (void*)half);
	pthread_create(&tid[1], NULL, thread_func, (void*)half);

	// get return value
	pthread_join(tid[0], (void**)&a);
	pthread_join(tid[1], (void**)&b);
	*sum = *a + *b;
//printf("%d %d\n%d %d\n", (int)arg, *a, (int)arg, *b);

	// free previous values
	free(a);
	free(b);

	return (void*)sum;
}

int main()
{
	struct timespec begin, end;
	char* filename = "./temp.txt"; 

	// open file
	f_read = fopen(filename, "r");

	// create thread
	clock_gettime(CLOCK_MONOTONIC, &begin);
	int *result = (thread_func((void*)MAX_PROCESSES));
	clock_gettime(CLOCK_MONOTONIC, &end);

	// print result
	long time = end.tv_sec - begin.tv_sec + end.tv_nsec - begin.tv_nsec;
	printf("value of thread : %d\n", *result);
	printf("%lf\n", (double)time/1000000000);

	free(result);
	fclose(f_read);

	return 0;
}
