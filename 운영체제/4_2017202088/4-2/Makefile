EXEC = D_recompile

CC = gcc

default:
	$(CC) -o $(EXEC) D_recompile.c
dynamic:
	$(CC) -DDYNAMIC -o $(EXEC) D_recompile.c
test2:
	$(CC) -o test2 D_recompile_test.c
objdump:
	$(CC) -c D_recompile_test.c

clean:
	rm -rf D_recompile $(EXEC)
