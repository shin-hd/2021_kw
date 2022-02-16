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
#define BUFFER_LENGTH 128

void **syscall_table;
void *real_ftrace;

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

asmlinkage pid_t process_tracer(pid_t trace_task)
{
	// process info struct
	struct task_struct *task;
	struct mm_struct *mm;
	struct vm_area_struct *vm;
	struct file *file;

	// file path buffer
	char buf[128];
	char *file_path;

	if(trace_task < 0)				// if pid < 0, return -1
		return -1;
	else if(trace_task == 0)			// if pid == 0, get current task struct
		task = get_current();
	else						// if pid > 0, get task struct of pid
		task = get_pid_task(find_get_pid(trace_task), PIDTYPE_PID);

	// print start message
	printk("######## Loaded files of a process '%s(%d)' in VM ########", task->comm, task->pid);

	// get mm struct, vm struct 
	mm = task->mm;
	vm = mm->mmap;

	// for each vm struct
	while(vm) {
		// get file struct from vm
		file = vm->vm_file;

		// vm is used for mmap
		if(file) {
			// print mem, code, data, heap address and full path
			memset(buf, '\0', BUFFER_LENGTH);
			file_path = dentry_path_raw(vm->vm_file->f_path.dentry, buf, BUFFER_LENGTH-1);
			printk(" mem(%x~%x) code(%x, %x) data(%x, %x) heap(%x, %x) %s\n",
				vm->vm_start, vm->vm_end, mm->start_code, mm->end_code, mm->start_data,
				mm->end_data, mm->start_brk, mm->brk, file_path);
		}

		// get next vm
		vm = vm->vm_next;
	}

	// print end message
	printk("################################################################");

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
