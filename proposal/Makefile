TARGET = description

#INCLUDES := test.tex

%.pdf: %.tex
	pdflatex $*
	-bibtex $*
	pdflatex $*
	pdflatex $*

$(TARGET).pdf: $(INCLUDES) *.tex *.bib *.sty


clean:
	rm -rf *.blg 
	rm -rf *.out 
	rm -rf *.bbl 
	rm -rf *.log
	rm -rf *.ind
	rm -rf *.ilg
	rm -rf *.lot
	rm -rf *.lof
	rm -rf *.ind
	rm -rf *.idx
	rm -rf *.aux
	rm -rf *.toc

clean-pdf: clean
	rm -f ${TARGET}.pdf

.DEFAULT_GOAL := $(TARGET).pdf