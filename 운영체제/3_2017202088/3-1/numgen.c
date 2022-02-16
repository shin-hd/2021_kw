#include <stdio.h>
#include <stdlib.h>

#define MAX_PROCESSES 64

int main()
{
	int i;
	char* filename = "./temp.txt";

	FILE* f_write = fopen(filename, "w");

	for(i = 0; i < MAX_PROCESSES * 2; i++)
	{
		fprintf(f_write, "%d\n", i+1);
	}

	fclose(f_write);

	return 0;
}
