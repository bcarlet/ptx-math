#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

#include "ptxm/models.h"

using model_map = std::map<std::string, float (*)(float)>;

static void usage [[noreturn]] (const char *prog_name, const model_map &functions)
{
    std::cerr << "Usage: " << prog_name << " <function> <args>...\n";
    std::cerr << "\nAvailable functions:\n";

    for (const auto &node : functions)
    {
        std::cerr << "  " << node.first << '\n';
    }

    std::exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    model_map functions;

    functions["rcp_sm5x"] = ptxm_rcp_sm5x;
    functions["sqrt_sm5x"] = ptxm_sqrt_sm5x;
    functions["sqrt_sm6x"] = ptxm_sqrt_sm6x;
    functions["rsqrt_sm5x"] = ptxm_rsqrt_sm5x;
    functions["sin_sm5x"] = ptxm_sin_sm5x;
    functions["sin_sm70"] = ptxm_sin_sm70;
    functions["cos_sm5x"] = ptxm_cos_sm5x;
    functions["cos_sm70"] = ptxm_cos_sm70;
    functions["lg2_sm5x"] = ptxm_lg2_sm5x;
    functions["ex2_sm5x"] = ptxm_ex2_sm5x;

    float (*model)(float);

    if (argc >= 2)
    {
        const auto lookup = functions.find(argv[1]);

        if (lookup != functions.end())
            model = lookup->second;
        else
            usage(argv[0], functions);
    }
    else
    {
        usage(argv[0], functions);
    }

    std::cout << std::showpoint << std::hexfloat;

    for (int i = 2; i < argc; i++)
    {
        float input;

        try {
            input = std::stof(argv[i]);
        } catch (...) {
            std::cerr << argv[i] << " (input error)\n";
            continue;
        }

        std::cout << input << ' ' << model(input) << '\n';
    }

    return EXIT_SUCCESS;
}
