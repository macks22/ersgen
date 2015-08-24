FILE=bpmlr

all:
	pdflatex $(FILE).tex
	bibtex   $(FILE).aux
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	rm -f $(FILE).aux
	rm -f $(FILE).bbl
	rm -f $(FILE).log
	rm -f $(FILE).blg
	rm -f texput.log

.PHONY: clean
