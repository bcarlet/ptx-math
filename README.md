# PTX-Math

PTXM provides bit-accurate implementations of the elementary function approximation instructions available in Nvidia's PTX ISA.

There are presently implementations for:

- `rcp.approx.f32`
- `sqrt.approx.f32`
- `rsqrt.approx.f32`
- `sin.approx.f32`
- `cos.approx.f32`
- `lg2.approx.f32`
- `ex2.approx.f32`

See the section below for architecture compatibility.

## Building

The minimum CMake version for building with Unix makefiles is 3.10. The Visual Studio generators require at least 3.11.

To build with a local installation:

    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=../install ..
    cmake --build . --config Release --target install

Available options:

- `-DBUILD_SHARED_LIBS=ON` to build as a shared library
- `-DPTXM_USE_PIC=ON` to generate position-independent code (automatic for shared libraries)
- `-DPTXM_ENABLE_CUDA=OFF` to disable targets requiring CUDA (namely `ptxmvalidate`)

Building requires compilers for C11 and C++11. CUDA is not required to build/use the main library and can be disabled with the provided option. The PTXM library requires support for the BMI2 extension.

## Compatibility

Provided are results of exhaustive tests with GPUs of various compute capability. The table indicates, for each function and architecture, whether the function produced values identical to those of the corresponding PTX instruction executed on a test GPU from the given architecture. These tests are exhaustive across all single-precision inputs.

|                   | sm_5x | sm_6x | sm_70 | sm_75 |
| :---              | :---: | :---: | :---: | :---: |
| `ptxm_rcp_sm5x`   |  Yes  |  Yes  |  Yes  |  Yes  |
| `ptxm_sqrt_sm5x`  |  Yes  |   -   |   -   |   -   |
| `ptxm_sqrt_sm6x`  |   -   |  Yes  |  Yes  |  Yes  |
| `ptxm_rsqrt_sm5x` |  Yes  |  Yes  |  Yes  |  Yes  |
| `ptxm_sin_sm5x`   |  Yes  |  Yes  |   -   |   -   |
| `ptxm_cos_sm5x`   |  Yes  |  Yes  |   -   |   -   |
| `ptxm_lg2_sm5x`   |  Yes  |  Yes  |  Yes  |  Yes  |
| `ptxm_ex2_sm5x`   |  Yes  |  Yes  |  Yes  |  Yes  |

These tests indicate expected compatibility. The `ptxmvalidate` utility can be used to verify correctness for a particular device.
