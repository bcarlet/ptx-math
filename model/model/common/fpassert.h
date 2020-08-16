#ifndef FPASSERT_H
#define FPASSERT_H

#include <assert.h>
#include <limits.h>

static_assert(CHAR_BIT == 8, "byte not 8 bits");
static_assert(sizeof(float) == 4, "float not 4 bytes");
static_assert(sizeof(double) == 8, "double not 8 bytes");

#endif
