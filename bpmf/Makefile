NUMPY=/home/mack/anaconda/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -O3 -I/usr/include/python2.7/ -I$(NUMPY)


cpmf: cpmf.pyx
	cython -a cpmf.pyx
	gcc -c $(CFLAGS) cpmf.c
	gcc -shared cpmf.o -o cpmf.so

clean:
	rm -f cpmf.c
	rm -f cpmf.o
	rm -f cpmf.html

PHONY: clean
