#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP

#include <chrono>

template<class Rep, class Period = std::ratio<1>>
class stopwatch
{
public:
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<Rep, Period>;

    /**
     * Construct and start a stopwatch.
     */
    stopwatch() :
        begin(clock::now())
    {
    }

    /**
     * Reset the elapsed time.
     */
    void reset()
    {
        begin = clock::now();
    }

    /**
     * Get the time elapsed since the stopwatch's creation or last reset.
     */
    Rep elapsed() const
    {
        const clock::time_point end = clock::now();
        const duration diff = std::chrono::duration_cast<duration>(end - begin);

        return diff.count();
    }

private:
    clock::time_point begin;
};

#endif
