#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>

#define MAX_PROCESSES 10000

int create_child()
{
	FILE* f_read;
	char filename[16] = "./temp/";
	char line[4] = {0, };
	pid_t pid;
	int i, num, sum = 0;

	for(i = 0; i < MAX_PROCESSES; i++)	// create child process
	{
		if((pid = fork()) < 0)	// fork error
		{
			printf("fork error\n");
			return 1;
		}
		else if(pid == 0)	// at child
		{
			// set prioirty
			struct sched_param param;
			param.sched_priority = (sched_get_priority_min(SCHED_RR) + sched_get_priority_max(SCHED_RR))/2;

			// SCHED_OTHER, SCHED_FIFO, SCHED_RR
			sched_setscheduler(0, SCHED_FIFO, &param);
			//nice(19);

			// open "./temp/i"
			sprintf(filename, "%s%d", filename, i);
			f_read = fopen(filename, "r");

			// read number
			fgets(line, 4, f_read);
			num = atoi(line);

			fclose(f_read);
			exit(num);
		}
	}

	for(i = 0; i < MAX_PROCESSES; i++)	// wait child processes
	{
		wait(&num);
		sum += num >> 8;
	}

	return sum;
}

int main()
{
	struct timespec begin, end;

	// do fork
	clock_gettime(CLOCK_MONOTONIC, &begin);
	int result = create_child();
	clock_gettime(CLOCK_MONOTONIC, &end);

	// print result
	long time = end.tv_sec - begin.tv_sec + end.tv_nsec - begin.tv_nsec;
	printf("value of fork : %d\n", result);
	printf("%lf\n", (double)time/1000000000);

	return 0;
}
