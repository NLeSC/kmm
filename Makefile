CLANG_FORMAT=clang-format-16 --verbose

pretty:
	${CLANG_FORMAT} -i include/kmm/*/*.hpp
	$(CLANG_FORMAT) -i src/*/*.cpp
	${CLANG_FORMAT} -i examples/*.cu
	${CLANG_FORMAT} -i test/*/*.cpp
	#${CLANG_FORMAT} -i benchmarks/*.cu

docs:
	cd docs && doxygen Doxyfile

all: pretty

.PHONY : pretty docs
