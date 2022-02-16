OBJS = numgen fork thread
CC = gcc

all: $(OBJS)

numgen: numgen.c
	$(CC) -o $@ $<

fork: fork.c
	$(CC) -o $@ $<

thread: thread.c
	$(CC) -pthread -o $@ $<

clean:
	rm numgen fork thread temp.txt
