#include "binsearch.hpp"

static uint64_t bin_mid(uint64_t l, uint64_t r)
{
    return l + (r - l) / 2u;
}

bs_state bin_search(uint64_t l, uint64_t r, uint64_t &test)
{
    if (l > r)
        return bs_state::FAIL;

    test = bin_mid(l, r);

    return bs_state::CONTINUE;
}

bs_state bin_search(uint64_t &l, uint64_t &r, uint64_t &test, int test_cmp)
{
    if (test_cmp < 0)
    {
        if (test == UINT64_MAX)
            return bs_state::FAIL;
        else
            l = test + 1u;
    }
    else if (test_cmp > 0)
    {
        if (test == 0u)
            return bs_state::FAIL;
        else
            r = test - 1u;
    }
    else
    {
        return bs_state::SUCCESS;
    }

    if (l > r)
        return bs_state::FAIL;

    test = bin_mid(l, r);

    return bs_state::CONTINUE;
}
