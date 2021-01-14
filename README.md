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

Note that compatibility varies with GPU architecture. Function aliases mapping to the correct implementations are defined for each supported architecture.

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

Building requires compilers for C11 and C++11. Components requiring CUDA are optional and can be disabled with the provided CMake option. The PTXM library requires processor support for the BMI2 x86 extension.

## Caveats

PTX assembly targets a virtual GPU architecture, and some PTX instructions may be compiled to multiple native device instructions. Consistency of PTX semantics is therefore dependent on consistent JIT compilation of PTX assembly to native (SASS) assembly by the GPU driver. Furthermore, SASS is architecture-dependent, and thus semantics are not necessarily preserved across architectures. The `ptxmvalidate` utility can be used to verify correctness of the PTXM implementations for a particular target device.

## Validation

The `ptxmvalidate` utility performs exhaustive simulation of the selected function across all single-precision inputs. Function output is compared to that of the corresponding PTX instruction executed on the selected GPU device.

When building, you may wish to specify a real target architecture (e.g., `sm_50`) rather than a virtual target architecture (e.g., `compute_50`) so that SASS is generated directly by `nvcc`. This bypasses the JIT compilation and allows for inspection of the resulting device code.

### Results

Provided are results of validation tests with GPUs of various compute capability. These tests indicate expected architecture compatibility.

|                   | `sm_5x` | `sm_6x` | `sm_70` | `sm_75` |
| :---              |  :---:  |  :---:  |  :---:  |  :---:  |
| `ptxm_rcp_sm5x`   |   Yes   |   Yes   |   Yes   |   Yes   |
| `ptxm_sqrt_sm5x`  |   Yes   |    -    |    -    |    -    |
| `ptxm_sqrt_sm6x`  |    -    |   Yes   |   Yes   |   Yes   |
| `ptxm_rsqrt_sm5x` |   Yes   |   Yes   |   Yes   |   Yes   |
| `ptxm_sin_sm5x`   |   Yes   |   Yes   |    -    |    -    |
| `ptxm_sin_sm70`   |    -    |    -    |   Yes   |   Yes   |
| `ptxm_cos_sm5x`   |   Yes   |   Yes   |    -    |    -    |
| `ptxm_cos_sm70`   |    -    |    -    |   Yes   |   Yes   |
| `ptxm_lg2_sm5x`   |   Yes   |   Yes   |   Yes   |   Yes   |
| `ptxm_ex2_sm5x`   |   Yes   |   Yes   |   Yes   |   Yes   |

Function aliases are set in accordance with this table.

## License

Licensed under the MIT License.
