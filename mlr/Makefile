NUMPY=/home/mack/anaconda/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -O3 -I/usr/include/python2.7/ -I$(NUMPY)


cmlr: cmlr.pyx
	cython -a cmlr.pyx
	gcc -c $(CFLAGS) cmlr.c
	gcc -shared cmlr.o -o cmlr.so

clean:
	rm -f cmlr.c
	rm -f cmlr.o
	rm -f cmlr.html

PHONY: clean
