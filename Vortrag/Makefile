all: build/main.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/main.pdf: FORCE | build
	latexmk $(TeXOptions) main.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
