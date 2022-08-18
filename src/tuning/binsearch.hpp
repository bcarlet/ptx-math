#ifndef TUNING_BINSEARCH_HPP
#define TUNING_BINSEARCH_HPP

#include <limits>
#include <type_traits>

namespace tuning
{

template<
    class T,
    class = typename std::enable_if<std::is_unsigned<T>::value>::type>
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
        m_left(left),
        m_right(right),
        m_test(left + (right - left) / 2u)
    {
    }

    T point() const
    {
        return m_test;
    }

    state step(int cmp)
    {
        if (cmp < 0)
        {
            if (m_test == std::numeric_limits<T>::max())
                return FAIL;
            else
                m_left = m_test + 1u;
        }
        else if (cmp > 0)
        {
            if (m_test == std::numeric_limits<T>::lowest())
                return FAIL;
            else
                m_right = m_test - 1u;
        }
        else
        {
            return SUCCESS;
        }

        if (m_left > m_right)
            return FAIL;

        m_test = m_left + (m_right - m_left) / 2u;

        return CONTINUE;
    }

private:
    T m_left, m_right, m_test;
};

}   // namespace tuning

#endif
