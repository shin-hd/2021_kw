#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/fcntl.h>
#include <linux/module.h>
#include <linux/highmem.h>
#include <linux/kallsyms.h>
#include <linux/syscalls.h>
#include <linux/string.h>
#include <linux/sched.h>
#include <linux/list.h>
#include <linux/types.h>
#include <asm/syscall_wrapper.h>
#include <asm/uaccess.h>

#define __NR_ftrace 336

void **syscall_table;
void *real_ftrace;

char* state_message = "- task state :";
char* end_message = "##### END OF INFORMATION #####";
int sibling_count;
int child_count;

// chmod syscall table
void make_rw(void *addr)
{
	unsigned int level;
	pte_t *pte = lookup_address((u64)addr, &level);

	if(pte->pte &~ _PAGE_RW)
		pte->pte |= _PAGE_RW;
}

// recover mod of syscall table
void make_ro(void *addr)
{
	unsigned int level;
	pte_t *pte = lookup_address((u64)addr, &level);

	pte->pte = pte->pte &~ _PAGE_RW;
}

void set_fork_count(void) {
	//struct task_with_fork *header;

	struct task_struct *task;
	for_each_process(task) {
	}
}

asmlinkage pid_t process_tracer(pid_t trace_task)
{
	struct task_struct *task, *leader_task, *sibling_task, *child_task;

	if(trace_task < 0)				// if pid < 0, return -1
		return -1;
	else if(trace_task == 0)			// if pid == 0, get current task struct
		task = get_current();
	else						// if pid > 0, get task struct of pid
		task = get_pid_task(find_get_pid(trace_task), PIDTYPE_PID);

	// print process name
	printk("##### TASK INFORMATION of ''[%d] %s'' #####", task->pid, task->comm);

	// print process state
	switch(task->state)
	{
		case TASK_RUNNING:
			printk("%s Running or ready\n", state_message);
			break;
		case TASK_UNINTERRUPTIBLE:
			printk("%s Wait with ignoring all signals\n", state_message);
			break;
		case TASK_INTERRUPTIBLE:
			printk("%s Wait\n", state_message);
			break;
		case __TASK_STOPPED:
			printk("%s Stopped\n", state_message);
			break;
		case EXIT_ZOMBIE:
			printk("%s Zombie process\n", state_message);
			break;
		case EXIT_DEAD:
			printk("%s Dead\n", state_message);
			break;
		default:
			printk("%s etc.\n", state_message);
			break;
	}

	// print process group leader
	leader_task = task->group_leader;
	printk("- Process Group Leader : [%d] %s\n", leader_task->pid, leader_task->comm);

	// print number of context switched
	printk("- Number of context switched : %lu\n", task->nivcsw);

	// print number of calling fork
	printk("- Number of calling fork() : \n");

	// print parent process
	printk("- it's parrent process : [%d] %s\n", task->parent->pid, task->parent->comm);

	// print sibling process(es)
	sibling_count = 0;
	printk("- it's sibling process : \n");
	list_for_each_entry(sibling_task, &task->sibling, sibling) {
		if(sibling_task->pid) {
			printk("  > [%d] %s\n", sibling_task->pid, sibling_task->comm);
			sibling_count++;
		}
	}
	if(sibling_count)
		printk("  > This process has %d sibling process(es)\n", sibling_count);
	else
		printk("  > It has no sibling.\n");

	// print child process(es)
	child_count = 0;
	printk("- it's child process(es) : \n");
	list_for_each_entry(child_task, &task->children, sibling) {
		printk("  > [%d] %s\n", child_task->pid, child_task->comm);
		child_count++;
	}
	if(child_count)
		printk("  > This process has %d child process(es)\n", child_count);
	else
		printk("  > It has no child.\n");

	printk(end_message);

	return task->pid;
}


__SYSCALL_DEFINEx(1, _process_tracer, pid_t, pid)
{
	return process_tracer(pid);
}

static int __init hooking_init(void)
{
	syscall_table = (void**) kallsyms_lookup_name("sys_call_table");

	make_rw(syscall_table);

	real_ftrace = syscall_table[__NR_ftrace];
	syscall_table[__NR_ftrace] = __x64_sys_process_tracer;

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
