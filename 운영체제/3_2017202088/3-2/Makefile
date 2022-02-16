OBJS = filegen schedtest
CC = gcc

all: $(OBJS)

filegen: filegen.c
	$(CC) -o $@ $<

schedtest: schedtest.c
	$(CC) -o $@ $<

clean:
	rm -rf $(OBJS) temp
