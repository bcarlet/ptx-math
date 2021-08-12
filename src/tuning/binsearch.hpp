#ifndef TUNING_BINSEARCH_HPP
#define TUNING_BINSEARCH_HPP

#include <limits>
#include <type_traits>

template<class T, class = typename std::enable_if<std::is_unsigned<T>::value>::type>
class binsearch
{
public:
    enum state
    {
        CONTINUE,
        SUCCESS,
        FAIL
    };

    binsearch(T left, T right) :
        left(left),
        right(right),
        test(left + (right - left) / 2u)
    {
    }

    T point() const
    {
        return test;
    }

    state step(int cmp)
    {
        if (cmp < 0)
        {
            if (test == std::numeric_limits<T>::max())
                return FAIL;
            else
                left = test + 1u;
        }
        else if (cmp > 0)
        {
            if (test == std::numeric_limits<T>::lowest())
                return FAIL;
            else
                right = test - 1u;
        }
        else
        {
            return SUCCESS;
        }

        if (left > right)
            return FAIL;

        test = left + (right - left) / 2u;

        return CONTINUE;
    }

private:
    T left, right, test;
};

#endif
