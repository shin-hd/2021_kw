#include "ftracehooking.h"

#define __NR_ftrace 336

void **syscall_table;

pid_t (*real_ftrace)(pid_t pid);

char end1[255] = {0, };
char end2[255] = {0, };
char end3[255] = {0, };

pid_t savedPid = 0;		// ftrace pid
char fname[255] = {0, };	// open file name
unsigned long rw[2] = {0, };	// read/write bytes
int fcount[5] = {0, };		// r/w/o/c/lseek count
// share with ioftracehooking.c
EXPORT_SYMBOL(savedPid);
EXPORT_SYMBOL(fname);
EXPORT_SYMBOL(rw);
EXPORT_SYMBOL(fcount);

__SYSCALL_DEFINEx(1, _ftrace, pid_t, pid)
{
	// finish ftrace
	if(savedPid > 0 && pid == 0)
	{
		struct task_struct* task;
		task = get_current();

		// set result message
		sprintf(end1, "[2017202088] /%s file[%s] stats[x] read - %lu / written - %lu", task->comm, fname, rw[0], rw[1]);
		sprintf(end2, "open[%d] close[%d] read[%d] write[%d] lseek[%d]", fcount[0], fcount[1], fcount[2], fcount[3], fcount[4]);
		sprintf(end3, "OS Assignment2 ftrace [%d] End\n", savedPid);

		// print result
		printk(end1);
		printk(end2);
		printk(end3);

		// reset
		savedPid = 0;
		fname[0] = '\0';
		fcount[0] = 0;
		fcount[1] = 0;
		fcount[2] = 0;
		fcount[3] = 0;
		fcount[4] = 0;
		rw[0] = 0;
		rw[1] = 0;
	}
	// start ftrace
	else if(pid > 0)
	{
		// save pid
		savedPid = pid;

		// print start message
		printk("OS Assignment2 ftrace [%d] Start\n", savedPid);
		
	}
	// negative pid
	else
	{
		printk("ftrace error: unexpected parameter [%d]\n", pid);
	}
	return pid;
}

static int __init hooking_init(void)
{
	syscall_table = (void**) kallsyms_lookup_name("sys_call_table");

	make_rw(syscall_table);

	real_ftrace = syscall_table[__NR_ftrace];
	syscall_table[__NR_ftrace] = __x64_sys_ftrace;

	return 0;
}

static void __exit hooking_exit(void)
{
	syscall_table[__NR_ftrace] = real_ftrace;

	make_ro(syscall_table);
}

module_init(hooking_init);
module_exit(hooking_exit);
MODULE_LICENSE("GPL");
