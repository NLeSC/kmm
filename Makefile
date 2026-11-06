CLANG_FORMAT=clang-format --verbose

pretty:
	find include src test examples \
		\( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -type f -print0 \
		| xargs -0 -r ${CLANG_FORMAT} -i

docs:
	cd docs && doxygen Doxyfile

all: pretty

.PHONY : pretty docs
