/*
	#include <> - others
	#include "" - my own

	// if referring to external variables outside function, must use extern keyword.
	int i = 0;
	void testFunc() {
	  extern int i; //i here is now using external variable i
}

struct rectangle {
  int width;
  int height;
};

typedef struct rectangle rect;

 */
#include <stdio.h>

void print_int_array(int* array, int length) {
	for (int i = 0; i < length; i++) {
		printf("%d, ", *(array + i));
	}

	printf("\n");
	fflush(stdout);
}

int increment(int x) {
	return x + 1;
}

void map_int(int (*function)(int), int* array, int length) {
	for (int i = 0; i < length; i++) {
		array[i] = function(array[i]);
	}
}

int main(int argc, char const *argv[]) {
	int length = 5;
	int numbas[length] = {1, 2, 3, 4, 5};
	map_int(increment, numbas, length);
	print_int_array(numbas, length);
	return 0;
}