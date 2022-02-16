#include <stdio.h>
#include <stdint.h> 
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/user.h>
#include <sys/mman.h>

int segment_id;

uint8_t* Operation;
uint8_t* compiled_code;

void sharedmem_init(); // 공유 메모리에 접근
void sharedmem_exit();
void drecompile_init(); // memory mapping 시작 
void drecompile_exit(); 
void* drecompile(uint8_t *func); //최적화하는 부분

int main(void)
{
	int (*func)(int a);
	clock_t start, end;

	sharedmem_init();
	drecompile_init();

	func = (int (*)(int a))drecompile(Operation);

	// test
	start = clock();
	for(int i = 0; i < 10000; i++) func(i);
	end = clock();

	// print result
	printf("total execution time: %lf sec\n", (double)(end-start)/CLOCKS_PER_SEC);

	drecompile_exit();
	sharedmem_exit();

	return 0;
}

void sharedmem_init()
{
	segment_id = shmget(1234, PAGE_SIZE, 0);
	Operation = (uint8_t*)shmat(segment_id, NULL, 0);
}

void sharedmem_exit()
{
	shmdt(Operation);
	shmctl(segment_id, IPC_RMID, NULL);
}

void drecompile_init(uint8_t *func)
{
	int fd;
	int i = 0;
	char temp[PAGE_SIZE];

	fd = open("Operation", O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IXUSR);
	for(i = 0; i < PAGE_SIZE; i++) temp[i] = '@';
	write(fd, temp, PAGE_SIZE);
	lseek(fd, 0, 0);

	// memory mapping
	compiled_code = mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	// copy memory
	memcpy(compiled_code, Operation, PAGE_SIZE);
}

void drecompile_exit()
{
	// unmap
	munmap(compiled_code, PAGE_SIZE);
}

void* drecompile(uint8_t* func)
{
#ifdef DYNAMIC
	int i = 0, j = 0;
	uint8_t code[3];

	// for all function instruction
	while(func[i] != 0xc3) {
		// operation, source register
		code[0] = func[i];

		// op is not target of optimization
		if(code[0] != 0x83 && code[0] != 0x6b && code[0] != 0xf6) {
			compiled_code[j++] = func[i++];
		}
		else {
			// init value
			code[1] = func[i+1];
			code[2] = code[0] == 0x83 ? 0 : 1;

			if(code[0] == 0xf6) {	// div
				// same code[0-1]
				while(func[i] == code[0] && func[i+1] == code[1]) {
					code[2] *= func[i-1];

					// if repeated div, i=i+4
					// else,	    i=i+2
					i = func[i+4] == code[0] ? i+4 : i+2;
				}
				compiled_code[j-1] = code[2];
				compiled_code[j++] = code[0];
				compiled_code[j++] = code[1];
			}
			else {			// not div
				// same code[0-1]
				while(func[i] == code[0] && func[i+1] == code[1]) {
					if(code[0] == 0x83) // add suv
						code[2] += func[i+2];
					else		    // imul
						code[2] *= func[i+2];
					i += 3;
				}
				compiled_code[j++] = code[0];
				compiled_code[j++] = code[1];
				compiled_code[j++] = code[2];
			}
		}
	}
	// end of func
	compiled_code[j] = func[i];
#endif

	// chmod r-x
	mprotect(compiled_code, PAGE_SIZE, PROT_READ | PROT_EXEC);

	return compiled_code;
}
