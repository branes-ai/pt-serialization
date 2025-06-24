# Direct PyTorch Model Parser with Inference

This project provides a C++ implementation for directly parsing PyTorch `.pt` files without requiring predefined FlatBuffer schemas (`.fbs` files). It extracts model parameters, analyzes the computation graph, executes inference, and provides comprehensive instrumentation.

## Key Features

- **Direct .pt File Parsing**: No need for predefined FlatBuffer schemas
- **Parameter Extraction**: Directly reads model weights and biases from the PyTorch model
- **Graph Analysis**: Parses the computation graph structure and operations
- **Inference Execution**: Runs forward pass with sample or custom inputs
- **Comprehensive Instrumentation**: Detailed profiling including timing, memory usage, and FLOP estimates
- **Model Analysis**: Generates detailed reports about model architecture and performance

## Key Changes from Original Files

### CMakeLists.txt Modifications
- **Removed**: All FlatBuffers dependencies, paths, and library linking
- **Removed**: Custom command for generating FlatBuffer headers from `.fbs` files
- **Removed**: `flow_graph_generated.h` generation
- **Changed**: Target name from `flow_graph_constructor` to `pytorch_parser`
- **Added**: Enhanced build configuration messages and optional optimizations
- **Simplified**: Now only requires LibTorch

### Files No Longer Needed
- `flow_graph.fbs` - FlatBuffer schema file
- `flow_graph_generated.h` - Auto-generated header
- Any FlatBuffers library files

### Build Script Compatibility
Your existing `build.bat` should work with the updated CMakeLists.txt, but the output executable name has changed to `pytorch_parser.exe`.

### Added Capabilities
- **Direct Parameter Access**: Extracts actual parameter values from the model
- **Inference Execution**: Can run forward pass with profiling
- **Enhanced Instrumentation**: Timing, memory tracking, and FLOP estimation
- **Flexible Input Handling**: Supports various input dimensions
- **Detailed Reporting**: Saves comprehensive analysis to text files

### New Components
- `InstrumentationData`: Tracks performance metrics
- `TensorInfo` & `ParameterInfo`: Stores tensor and parameter details
- `DirectPyTorchParser`: Main parser class with inference capabilities
- Memory usage tracking and FLOP estimation

## Requirements

- PyTorch C++ API (LibTorch)
- CMake 3.18 or later
- C++17 or later
- CUDA (optional, for GPU inference)

## Building

The CMakeLists.txt has been updated to remove FlatBuffers dependencies and only requires LibTorch.

**Important**: If upgrading from the previous version, clean your build directory first:

```bash
rm -rf build  # Linux/Mac
# or
rmdir /s build  # Windows
```

### Linux/Mac Build
```bash
mkdir build
cd build
cmake ..
make
```

### Windows Build
```cmd
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Using build.bat (Windows)
```cmd
build.bat
```

### Build Configuration
- **Target Name**: Changed from `flow_graph_constructor` to `pytorch_parser`
- **Dependencies**: Only LibTorch (FlatBuffers removed)
- **Output**: Executable named `pytorch_parser`

## Usage

### Basic Model Analysis
```bash
./pytorch_parser model.pt
```

### With Inference
```bash
./pytorch_parser model.pt --inference
```

### With Detailed Profiling
```bash
./pytorch_parser model.pt --inference --profile
```

### Windows Usage
```cmd
pytorch_parser.exe model.pt --inference --profile
```

## Command Line Options

- `--inference`: Execute inference with a sample input tensor
- `--profile`: Enable detailed profiling during inference (implies --inference)

## Sample Input Configuration

The default sample input is created as:
```cpp
auto sample_input = torch::randn({1, 3, 224, 224}); // Batch=1, Channels=3, Height=224, Width=224
```

**Important**: Modify the input dimensions in `main()` to match your model's expected input shape.

### Common Input Shapes
- **Vision Models**: `{1, 3, 224, 224}` (ImageNet standard)
- **MNIST**: `{1, 1, 28, 28}`
- **CIFAR-10**: `{1, 3, 32, 32}`
- **Text Models**: `{1, sequence_length}` or `{batch_size, sequence_length}`

## Output Files

### model_analysis.txt
Contains detailed model information including:
- Parameter names, shapes, and element counts
- Graph structure with node types and operations
- FLOP estimates and parameter counts per layer

### Console Output
- Model loading status
- Parameter extraction summary
- Graph parsing results
- Inference execution times
- Instrumentation results (if profiling enabled)

## Example Output

```
Direct PyTorch Model Parser with Inference
================================================================================
[LOADING PYTORCH MODEL]
File: resnet18.pt
[SUCCESS] PyTorch model loaded
Graph nodes: 25

[EXTRACTING MODEL PARAMETERS]
Parameter: conv1.weight Shape: [64, 3, 7, 7] Elements: 9408
Parameter: conv1.bias Shape: [64] Elements: 64
...
Total parameters extracted: 42

[PARSING COMPUTATION GRAPH]
Parsed: Conv2D_1 (aten::conv2d) Params: 9472
Parsed: ReLU_2 (aten::relu) Params: 0
...

[EXECUTING INFERENCE]
Input shape: [1, 3, 224, 224]
[SUCCESS] Inference completed in 45.23 ms
Output shape: [1, 1000]

[INSTRUMENTATION RESULTS]
Operation: forward_pass
  Duration: 45.230 ms
  Memory Delta: 12 MB
  Estimated FLOPs: 1814073344
```

## Advanced Usage

### Custom Input Tensor
To use a custom input tensor, modify the code in `main()`:

```cpp
// Replace the sample input creation with your custom tensor
auto custom_input = torch::ones({1, 3, 256, 256}); // Your custom shape
auto output = parser.executeInference(custom_input, enable_profiling);
```

### GPU Inference
Ensure your model and input tensors are on the same device:

```cpp
// Move model to GPU (if available)
if (torch::cuda::is_available()) {
    module.to(torch::kCUDA);
    sample_input = sample_input.to(torch::kCUDA);
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Ensure the `.pt` file is a TorchScript model (`torch.jit.trace()` or `torch.jit.script()`)
   - Check file path and permissions

2. **Input Shape Mismatch**
   - Verify the sample input shape matches your model's expected input
   - Check model documentation or original training code

3. **Memory Issues**
   - Large models may require significant RAM
   - Consider running without profiling for memory-constrained systems

4. **CUDA Errors**
   - Ensure PyTorch was built with CUDA support
   - Check GPU memory availability

### Performance Notes

- **Memory Tracking**: Currently simplified; enhance for production use
- **FLOP Estimation**: Provides estimates for common operations; may need refinement for complex models
- **Profiling Overhead**: Detailed profiling adds execution time

## Extension Points

The parser can be extended to support:
- Additional PyTorch operations in `getNodeType()`
- More sophisticated FLOP calculations in `estimateNodeComplexity()`
- Custom profiling metrics in `InstrumentationData`
- Model optimization analysis
- Quantization information extraction

## Compatibility

- **PyTorch Versions**: Tested with PyTorch 1.8+
- **Models**: Supports TorchScript models (.pt files created with `torch.jit`)
- **Platforms**: Linux, Windows, macOS (with appropriate LibTorch builds)

## Contributing

When extending the parser:
1. Add new node types to the `NodeType` enum
2. Update `getNodeType()` and `getNodeTypeName()` functions
3. Enhance `estimateNodeComplexity()` for new operations
4. Add appropriate parameter association logic in `associateParametersWithNode()`

## License

[Include your project's license information here]