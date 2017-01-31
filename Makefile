CFLAGS= -std=c99 -pedantic -Wall -march=native -O3
LIBS= 

prefixsum: prefixsum.c
	$(CC) prefixsum.c -o prefixsum $(CFLAGS) $(LIBS)

run: prefixsum
	./prefixsum

clean:
	rm -f prefixsum

.PHONY: run clean
