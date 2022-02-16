#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define MAX_PROCESSES 64

int read_and_add(int base)
{
	int a = 0, b = 0;
	char line[8] = {0, };
	long int offset;

	// open file
	FILE* f_read = fopen("./temp.txt", "r");

	// get offset from base
	if(base < 5) 		offset = base * 4;
	else if(base < 50)	offset = 21 + (base-5) * 6;
	else 			offset = 28 + 44*6 + (base-50)*8;

	// shift 
	fseek(f_read, offset, 0);

	// get numbers of file
	fgets(line, 8, f_read);
	a = atoi(line);
	fgets(line, 8, f_read);
	b = atoi(line);

	fclose(f_read);
	return a+b;
}

int create_child(int p, int base)
{
	// at leaves, read file and add
	if(p==1) {
		exit(read_and_add(base));
	}

	int n = 0, sum = 0;
	pid_t pid1, pid2;

	// create child 1
	pid1 = fork();
	if(pid1 < 0) {
		printf("fork error \n");
		exit(0);
	} else if(pid1 == 0) {	// at child process
		// reculsive call and return state to parent
		exit(create_child(p/2, base));
	}

	// create child 2
	pid2 = fork();
	if(pid2 < 0) {
		printf("fork error \n");
	} else if(pid2 == 0) {	// at child process
		// reculsive call and return state to parent
		exit(create_child(p/2, (p/2)+base));
	}

	// combine result of children
	waitpid(pid1, &n, 0);
	sum = n >> 8;
	waitpid(pid2, &n, 0);
	sum += n >> 8;

	return sum;
}

int main()
{
	struct timespec begin, end;

	// do fork
	clock_gettime(CLOCK_MONOTONIC, &begin);
	int result = create_child(MAX_PROCESSES, 0);
	clock_gettime(CLOCK_MONOTONIC, &end);

	// print result
	long time = end.tv_sec - begin.tv_sec + end.tv_nsec - begin.tv_nsec;
	printf("value of fork : %d\n", result);
	printf("%lf\n", (double)time/1000000000);

	return 0;
}
