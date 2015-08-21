NUMPY=/home/mack/anaconda/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -O3 -I/usr/include/python2.7/ -I$(NUMPY)
FILE=cfm


all: cfm

cfm: $(FILE).pyx
	cython -a $(FILE).pyx
	gcc -c $(CFLAGS) $(FILE).c
	gcc -shared $(FILE).o -o $(FILE).so

clean:
	rm -f $(FILE).c
	rm -f $(FILE).o
	rm -f $(FILE).html

PHONY: clean
