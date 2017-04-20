DIST=/home/msweene2/anaconda
PYHOME=$(DIST)/include/python2.7/
NUMPY=$(DIST)/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -ffast-math -O3 -I$(PYHOME) -I$(NUMPY)

IPR=cipr

ipr: $(IPR).pyx
	cython -a $(IPR).pyx
	gcc -c $(CFLAGS) $(IPR).c
	gcc -shared $(IPR).o -o $(IPR).so

clean:
	rm -f $(IPR).c
	rm -f $(IPR).o
	rm -f $(IPR).html


PHONY: clean
