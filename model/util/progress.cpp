#include "progress.hpp"

#include <cstdio>

static constexpr int PROGRESS_BAR_WIDTH = 60;

void print_progress_bar(float progress, const char *prefix)
{
    const int chars = int(progress * PROGRESS_BAR_WIDTH);
    const int percentage = int(progress * 100.0f);

    fprintf(stdout, "\r%s", prefix);

    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
    {
        fputc((i < chars) ? '#' : ' ', stdout);
    }

    fprintf(stdout, "[%d%%]", percentage);
    fflush(stdout);
}
