#ifndef TALLY_HPP
#define TALLY_HPP

#include <algorithm>
#include <limits>

template<class T>
class kahan_sum
{
public:
    void accumulate(T val)
    {
        const T y = val - c;
        const T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    T sum = 0;

private:
    T c = 0;
};

template<class T, class AvgType = double>
class tally
{
public:
    void accumulate(T val)
    {
        count++;
        min = std::min(min, val);
        max = std::max(max, val);
        sum.accumulate(val);
    }

    AvgType average() const
    {
        return (AvgType)sum.sum / count;
    }

    unsigned long long count = 0u;
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();

private:
    kahan_sum<T> sum;
};

#endif
