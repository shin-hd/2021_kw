#include "ftracehooking.h"

#define __NR_read 0
#define __NR_write 1
#define __NR_open 2
#define __NR_close 3
#define __NR_lseek 8

void **syscall_table;

asmlinkage ssize_t (*real_read)(unsigned int, char __user*, size_t) = NULL;
asmlinkage ssize_t (*real_write)(unsigned int, char __user*, size_t) = NULL;
asmlinkage long (*real_open)(const char __user*, int, umode_t) = NULL;
asmlinkage void (*real_close)(unsigned int) = NULL;
asmlinkage off_t (*real_lseek)(unsigned int, off_t, unsigned int) = NULL;

extern pid_t savedPid;
extern char fname[255];
extern unsigned long rw[2];
extern int fcount[5];

asmlinkage long ftrace_read(unsigned int fd, char __user* buf, size_t count)
{
	// if current pid is tracing process's pid
	struct task_struct* task;
	task = get_current();
	if(task->pid == savedPid)
	{
		// read count + 1
		fcount[2] += 1;

		// + read bytes
		rw[0] += count;
	}

	return real_read(fd, buf, count);
}

asmlinkage long ftrace_write(unsigned int fd, char __user* buf, size_t count)
{
	// if current pid is tracing process's pid
	struct task_struct* task;
	task = get_current();
	if(task->pid == savedPid) 
	{
		// write count + 1
		fcount[3] += 1;

		// + write bytes
		rw[1] += count;
	}

	return real_write(fd, buf, count);
}

asmlinkage long ftrace_open(const char __user* filename, int flags, umode_t mode)
{
	// if current pid is tracing process's pid
	struct task_struct* task;
	task = get_current();
	if(task->pid == savedPid)
	{
		// open count + 1
		fcount[0] += 1;
		// copy filename
		copy_from_user(fname, filename, 255);
	}

	return real_open(filename, flags, mode);
}

asmlinkage void ftrace_close(unsigned int fd)
{
	// if current pid is tracing process's pid, close count + 1
	struct task_struct* task;
	task = get_current();
	if(task->pid == savedPid) fcount[1] += 1;

	real_close(fd);
}

asmlinkage long ftrace_lseek(unsigned int fd, off_t offset, unsigned int whence) 
{
	// if current pid is tracing process's pid, lseek count + 1
	struct task_struct* task;
	task = get_current();
	if(task->pid == savedPid) fcount[4] += 1;

	return real_lseek(fd, offset, whence);
}

static int __init hooking_init(void)
{
	syscall_table = (void**) kallsyms_lookup_name("sys_call_table");

	make_rw(syscall_table);

	real_read = syscall_table[__NR_read];
	syscall_table[__NR_read] = ftrace_read; //__x64_sys_ftrace_read;

	real_write = syscall_table[__NR_write];
	syscall_table[__NR_write] = ftrace_write; //__x64_sys_ftrace_write;

	real_open = syscall_table[__NR_open];
	syscall_table[__NR_open] = ftrace_open; //__x64_sys_ftrace_open;

	real_close = syscall_table[__NR_close];
	syscall_table[__NR_close] = ftrace_close; //__x64_sys_ftrace_close;

	real_lseek = syscall_table[__NR_lseek];
	syscall_table[__NR_lseek] = ftrace_lseek; //__x64_sys_ftrace_lseek;

	return 0;
}

static void __exit hooking_exit(void)
{
	syscall_table[__NR_read] = real_read;
	syscall_table[__NR_write] = real_write;
	syscall_table[__NR_open] = real_open;
	syscall_table[__NR_close] = real_close;
	syscall_table[__NR_lseek] = real_lseek;

	make_ro(syscall_table);
}

module_init(hooking_init);
module_exit(hooking_exit);
MODULE_LICENSE("GPL");
