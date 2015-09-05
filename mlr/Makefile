NUMPY=/home/mack/anaconda/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -O3 -I/usr/include/python2.7/ -I$(NUMPY)
IPR=cipr
MLR=cmlr


mlr: $(MLR).pyx
	cython -a $(MLR).pyx
	gcc -c $(CFLAGS) $(MLR).c
	gcc -shared $(MLR).o -o $(MLR).so

ipr: $(IPR).pyx
	cython -a $(IPR).pyx
	gcc -c $(CFLAGS) $(IPR).c
	gcc -shared $(IPR).o -o $(IPR).so

clean:
	rm -f $(MLR).c
	rm -f $(MLR).o
	rm -f $(MLR).html
	rm -f $(IPR).c
	rm -f $(IPR).o
	rm -f $(IPR).html


PHONY: clean
