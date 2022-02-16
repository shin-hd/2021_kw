#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>

#define MAX_PROCESSES 10000

int main()
{
	int i;
	char path[16] = "./temp";

	// make directory
	mkdir(path, 0775);

	for(i = 0; i < MAX_PROCESSES; i++)
	{
		// ./temp/0~MAX_PROCESSES
		sprintf(path, "%s/%d", path, i); 

		// create file and write
		FILE* f_write = fopen(path, "w");
		fprintf(f_write, "%d", 1+rand()%9);
		fclose(f_write);

		// reset path
		path[6] = '\0';
	}

	return 0;
}
