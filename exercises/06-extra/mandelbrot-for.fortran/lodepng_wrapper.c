#include "lib/lodepng.h"

void lodepng_encode32_file_(const char *name, unsigned char *image, unsigned int *width, unsigned int *height) {
	lodepng_encode32_file(name, image, *width, *height);
}
