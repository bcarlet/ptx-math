#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP

#include <chrono>

template<class Duration = std::chrono::steady_clock::duration>
class stopwatch
{
public:
    using clock = std::chrono::steady_clock;

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
    typename Duration::rep elapsed() const
    {
        const std::chrono::time_point<clock> end = clock::now();
        const Duration diff = std::chrono::duration_cast<Duration>(end - begin);

        return diff.count();
    }

private:
    std::chrono::time_point<clock> begin;
};

#endif
