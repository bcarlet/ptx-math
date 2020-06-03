#ifndef FPASSERT_H
#define FPASSERT_H

#include <assert.h>
#include <limits.h>
#include <float.h>

static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
static_assert(sizeof(float) == 4, "sizeof(float) != 4");
static_assert(FLT_RADIX == 2, "FLT_RADIX != 2");

#endif
