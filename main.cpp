#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <memory>
#include <iomanip>
#include <sstream>

// Forward declarations
enum class NodeType {
    INPUT, OUTPUT, CONV2D, LINEAR, RELU, MAXPOOL2D, BATCHNORM2D, 
    DROPOUT, FLATTEN, ADD, MULTIPLY, SOFTMAX, LOGSOFTMAX, 
    ADAPTIVE_AVG_POOL2D, UNKNOWN
};

// Instrumentation data structure
struct InstrumentationData {
    std::string operation_name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    size_t memory_before_mb = 0;
    size_t memory_after_mb = 0;
    size_t flops_estimate = 0;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    
    double duration_ms() const {
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
    size_t memory_delta_mb() const {
        return memory_after_mb > memory_before_mb ? memory_after_mb - memory_before_mb : 0;
    }
};

// Tensor information structure
struct TensorInfo {
    std::vector<int64_t> shape;
    std::string dtype;
    size_t numel() const {
        size_t total = 1;
        for (auto dim : shape) total *= dim;
        return total;
    }
    
    size_t memory_bytes() const {
        return numel() * 4; // Assuming float32
    }
};

// Parameter information
struct ParameterInfo {
    std::string name;
    TensorInfo tensor_info;
    std::vector<float> data; // Store actual parameter values
};

// Node information structure
struct NodeInfo {
    std::string id;
    std::string name;
    NodeType type;
    std::string pytorch_op;
    std::vector<TensorInfo> input_shapes;
    std::vector<TensorInfo> output_shapes;
    std::map<std::string, std::string> attributes;
    std::vector<ParameterInfo> parameters;
    size_t flops_estimate = 0;
    
    size_t total_parameters() const {
        size_t total = 0;
        for (const auto& param : parameters) {
            total += param.tensor_info.numel();
        }
        return total;
    }
    
    size_t memory_bytes() const {
        size_t total = 0;
        for (const auto& param : parameters) {
            total += param.tensor_info.memory_bytes();
        }
        return total;
    }
};

// Helper functions
NodeType getNodeType(const c10::Symbol& kind) {
    std::string kind_str = kind.toQualString();
    
    if (kind_str == "aten::conv2d") return NodeType::CONV2D;
    if (kind_str == "aten::linear" || kind_str == "aten::addmm") return NodeType::LINEAR;
    if (kind_str == "aten::relu" || kind_str == "aten::relu_") return NodeType::RELU;
    if (kind_str == "aten::max_pool2d") return NodeType::MAXPOOL2D;
    if (kind_str == "aten::batch_norm") return NodeType::BATCHNORM2D;
    if (kind_str == "aten::dropout") return NodeType::DROPOUT;
    if (kind_str == "aten::flatten") return NodeType::FLATTEN;
    if (kind_str == "aten::add" || kind_str == "aten::add_") return NodeType::ADD;
    if (kind_str == "aten::mul" || kind_str == "aten::mul_") return NodeType::MULTIPLY;
    if (kind_str == "aten::softmax") return NodeType::SOFTMAX;
    if (kind_str == "aten::log_softmax") return NodeType::LOGSOFTMAX;
    if (kind_str == "aten::adaptive_avg_pool2d") return NodeType::ADAPTIVE_AVG_POOL2D;
    
    return NodeType::UNKNOWN;
}

std::string getNodeTypeName(NodeType type) {
    switch (type) {
        case NodeType::INPUT: return "Input";
        case NodeType::CONV2D: return "Conv2D";
        case NodeType::LINEAR: return "Linear";
        case NodeType::RELU: return "ReLU";
        case NodeType::MAXPOOL2D: return "MaxPool2D";
        case NodeType::BATCHNORM2D: return "BatchNorm2D";
        case NodeType::DROPOUT: return "Dropout";
        case NodeType::FLATTEN: return "Flatten";
        case NodeType::ADD: return "Add";
        case NodeType::MULTIPLY: return "Multiply";
        case NodeType::SOFTMAX: return "Softmax";
        case NodeType::LOGSOFTMAX: return "LogSoftmax";
        case NodeType::ADAPTIVE_AVG_POOL2D: return "AdaptiveAvgPool2D";
        case NodeType::OUTPUT: return "Output";
        default: return "Unknown";
    }
}

size_t getCurrentMemoryUsageMB() {
    // Simplified memory tracking - returns 0 for compatibility
    // In production, implement platform-specific memory tracking
    return 0;
}

// Enhanced PyTorch Model Parser with direct .pt parsing
class DirectPyTorchParser {
private:
    torch::jit::script::Module module;
    std::shared_ptr<torch::jit::Graph> graph;
    std::vector<NodeInfo> nodes;
    std::vector<InstrumentationData> profiling_data;
    std::map<std::string, ParameterInfo> model_parameters;
    
public:
    DirectPyTorchParser(const std::string& model_path) {
        std::cout << "[LOADING PYTORCH MODEL]" << std::endl;
        std::cout << "File: " << model_path << std::endl;
        
        try {
            module = torch::jit::load(model_path);
            module.eval();
            
            // Get the forward method graph
            graph = module.get_method("forward").graph();
            
            std::cout << "[SUCCESS] PyTorch model loaded" << std::endl;
            std::cout << "Graph nodes: " << std::distance(graph->nodes().begin(), graph->nodes().end()) << std::endl;
            
            // Extract model parameters directly
            extractModelParameters();
            
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load PyTorch model: " + std::string(e.what()));
        }
    }
    
    void extractModelParameters() {
        std::cout << "\n[EXTRACTING MODEL PARAMETERS]" << std::endl;
        
        auto named_parameters = module.named_parameters(true);
        for (const auto& param : named_parameters) {
            ParameterInfo param_info;
            param_info.name = param.name;
            
            auto tensor = param.value;
            auto sizes = tensor.sizes();
            param_info.tensor_info.shape = std::vector<int64_t>(sizes.begin(), sizes.end());
            param_info.tensor_info.dtype = "float32"; // Simplified
            
            // Extract actual parameter data - handle multi-dimensional tensors
            if (tensor.dtype() == torch::kFloat32) {
                // Flatten the tensor to 1D for data extraction
                auto flattened = tensor.flatten();
                param_info.data.resize(flattened.numel());
                std::memcpy(param_info.data.data(), flattened.data_ptr<float>(), 
                           flattened.numel() * sizeof(float));
            }
            
            model_parameters[param.name] = param_info;
            
            std::cout << "Parameter: " << param.name 
                      << " Shape: [";
            for (size_t i = 0; i < param_info.tensor_info.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << param_info.tensor_info.shape[i];
            }
            std::cout << "] Elements: " << param_info.tensor_info.numel() << std::endl;
        }
        
        std::cout << "Total parameters extracted: " << model_parameters.size() << std::endl;
    }
    
    void parseGraphStructure() {
        std::cout << "\n[PARSING COMPUTATION GRAPH]" << std::endl;
        
        int node_counter = 0;
        
        // Parse each node in the graph
        for (auto node : graph->nodes()) {
            NodeInfo node_info;
            node_info.id = "node_" + std::to_string(node_counter++);
            node_info.type = getNodeType(node->kind());
            node_info.name = getNodeTypeName(node_info.type) + "_" + std::to_string(node_counter);
            node_info.pytorch_op = node->kind().toQualString();
            
            // Parse input/output shapes
            parseNodeShapes(node, node_info);
            
            // Parse attributes
            parseNodeAttributes(node, node_info);
            
            // Estimate computational complexity
            estimateNodeComplexity(node_info);
            
            // Associate parameters with nodes
            associateParametersWithNode(node_info);
            
            nodes.push_back(node_info);
            
            std::cout << "Parsed: " << node_info.name 
                      << " (" << node_info.pytorch_op 
                      << ") Params: " << node_info.total_parameters() << std::endl;
        }
        
        std::cout << "Graph parsing complete: " << nodes.size() << " nodes" << std::endl;
    }
    
private:
    void parseNodeShapes(torch::jit::Node* node, NodeInfo& node_info) {
        // Extract input shapes
        for (auto input : node->inputs()) {
            if (auto tensor_type = input->type()->cast<c10::TensorType>()) {
                if (tensor_type->sizes().concrete_sizes()) {
                    TensorInfo tensor_info;
                    auto sizes = *tensor_type->sizes().concrete_sizes();
                    tensor_info.shape = std::vector<int64_t>(sizes.begin(), sizes.end());
                    tensor_info.dtype = "float32";
                    node_info.input_shapes.push_back(tensor_info);
                }
            }
        }
        
        // Extract output shapes
        for (auto output : node->outputs()) {
            if (auto tensor_type = output->type()->cast<c10::TensorType>()) {
                if (tensor_type->sizes().concrete_sizes()) {
                    TensorInfo tensor_info;
                    auto sizes = *tensor_type->sizes().concrete_sizes();
                    tensor_info.shape = std::vector<int64_t>(sizes.begin(), sizes.end());
                    tensor_info.dtype = "float32";
                    node_info.output_shapes.push_back(tensor_info);
                }
            }
        }
    }
    
    void parseNodeAttributes(torch::jit::Node* node, NodeInfo& node_info) {
        // Extract specific attributes based on operation type
        if (node_info.pytorch_op == "aten::conv2d") {
            node_info.attributes["operation"] = "convolution";
            // Try to extract kernel size, stride, etc. from the node
        }
        else if (node_info.pytorch_op == "aten::linear") {
            node_info.attributes["operation"] = "linear";
        }
        
        // Store the original PyTorch operation
        node_info.attributes["pytorch_op"] = node_info.pytorch_op;
    }
    
    void estimateNodeComplexity(NodeInfo& node_info) {
        // Estimate FLOPs based on operation type and tensor shapes
        if (node_info.type == NodeType::CONV2D && !node_info.input_shapes.empty() && !node_info.output_shapes.empty()) {
            // Simplified FLOP calculation for convolution
            auto input_shape = node_info.input_shapes[0].shape;
            auto output_shape = node_info.output_shapes[0].shape;
            
            if (input_shape.size() >= 4 && output_shape.size() >= 4) {
                size_t kernel_size = 3; // Assume 3x3 kernel
                size_t flops = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * 
                              input_shape[1] * kernel_size * kernel_size;
                node_info.flops_estimate = flops;
            }
        }
        else if (node_info.type == NodeType::LINEAR && !node_info.input_shapes.empty() && !node_info.output_shapes.empty()) {
            // FLOP calculation for linear layer
            auto input_shape = node_info.input_shapes[0].shape;
            auto output_shape = node_info.output_shapes[0].shape;
            
            if (input_shape.size() >= 2 && output_shape.size() >= 2) {
                size_t flops = input_shape[input_shape.size()-1] * output_shape[output_shape.size()-1];
                node_info.flops_estimate = flops;
            }
        }
    }
    
    void associateParametersWithNode(NodeInfo& node_info) {
        // Simple parameter association based on node type and naming patterns
        for (const auto& param_pair : model_parameters) {
            const std::string& param_name = param_pair.first;
            
            // Check if parameter name suggests association with this node type
            if ((node_info.type == NodeType::CONV2D && param_name.find("conv") != std::string::npos) ||
                (node_info.type == NodeType::LINEAR && param_name.find("linear") != std::string::npos) ||
                (node_info.type == NodeType::BATCHNORM2D && param_name.find("bn") != std::string::npos)) {
                
                node_info.parameters.push_back(param_pair.second);
            }
        }
    }
    
public:
    torch::Tensor executeInference(const torch::Tensor& input, bool enable_profiling = true) {
        std::cout << "\n[EXECUTING INFERENCE]" << std::endl;
        std::cout << "Input shape: [";
        for (int i = 0; i < input.dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input.size(i);
        }
        std::cout << "]" << std::endl;
        
        if (enable_profiling) {
            profiling_data.clear();
        }
        
        auto overall_start = std::chrono::high_resolution_clock::now();
        
        torch::Tensor result;
        
        if (enable_profiling) {
            // Profile each operation during inference
            result = executeWithProfiling(input);
        } else {
            // Simple inference without detailed profiling
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            result = module.forward(inputs).toTensor();
        }
        
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double, std::milli>(overall_end - overall_start).count();
        
        std::cout << "[SUCCESS] Inference completed in " << total_time << " ms" << std::endl;
        std::cout << "Output shape: [";
        for (int i = 0; i < result.dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << result.size(i);
        }
        std::cout << "]" << std::endl;
        
        return result;
    }
    
private:
    torch::Tensor executeWithProfiling(const torch::Tensor& input) {
        // This is a simplified profiling approach
        // In practice, you'd want to hook into PyTorch's profiling APIs
        
        InstrumentationData overall_profile;
        overall_profile.operation_name = "forward_pass";
        overall_profile.start_time = std::chrono::high_resolution_clock::now();
        overall_profile.memory_before_mb = getCurrentMemoryUsageMB();
        
        // Convert input shape
        auto input_sizes = input.sizes();
        overall_profile.input_shapes.push_back(std::vector<int64_t>(input_sizes.begin(), input_sizes.end()));
        
        // Execute the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto result = module.forward(inputs).toTensor();
        
        overall_profile.end_time = std::chrono::high_resolution_clock::now();
        overall_profile.memory_after_mb = getCurrentMemoryUsageMB();
        
        // Convert output shape
        auto output_sizes = result.sizes();
        overall_profile.output_shapes.push_back(std::vector<int64_t>(output_sizes.begin(), output_sizes.end()));
        
        // Estimate total FLOPs
        overall_profile.flops_estimate = estimateTotalFLOPs();
        
        profiling_data.push_back(overall_profile);
        
        return result;
    }
    
    size_t estimateTotalFLOPs() {
        size_t total_flops = 0;
        for (const auto& node : nodes) {
            total_flops += node.flops_estimate;
        }
        return total_flops;
    }
    
public:
    void printModelSummary() {
        std::cout << "\n[MODEL SUMMARY]" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Count node types
        std::map<NodeType, int> type_counts;
        for (const auto& node : nodes) {
            type_counts[node.type]++;
        }
        
        std::cout << "Model Architecture:" << std::endl;
        for (const auto& pair : type_counts) {
            std::cout << "  " << getNodeTypeName(pair.first) << ": " << pair.second << std::endl;
        }
        
        // Calculate total statistics
        size_t total_params = 0;
        size_t total_memory = 0;
        size_t total_flops = 0;
        
        for (const auto& param_pair : model_parameters) {
            total_params += param_pair.second.tensor_info.numel();
            total_memory += param_pair.second.tensor_info.memory_bytes();
        }
        
        for (const auto& node : nodes) {
            total_flops += node.flops_estimate;
        }
        
        std::cout << "\nModel Statistics:" << std::endl;
        std::cout << "  Total Parameters: " << total_params << std::endl;
        std::cout << "  Memory Footprint: " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Estimated FLOPs: " << total_flops << std::endl;
        std::cout << "  Total Nodes: " << nodes.size() << std::endl;
        
        // Detailed parameter breakdown
        std::cout << "\nDetailed Parameter Information:" << std::endl;
        for (const auto& param_pair : model_parameters) {
            const auto& param = param_pair.second;
            std::cout << "  " << param.name << ": ";
            for (size_t i = 0; i < param.tensor_info.shape.size(); ++i) {
                if (i > 0) std::cout << "x";
                std::cout << param.tensor_info.shape[i];
            }
            std::cout << " (" << param.tensor_info.numel() << " params)" << std::endl;
        }
    }
    
    void printInstrumentationResults() {
        if (profiling_data.empty()) {
            std::cout << "\n[NO PROFILING DATA AVAILABLE]" << std::endl;
            return;
        }
        
        std::cout << "\n[INSTRUMENTATION RESULTS]" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        for (const auto& profile : profiling_data) {
            std::cout << "Operation: " << profile.operation_name << std::endl;
            std::cout << "  Duration: " << std::fixed << std::setprecision(3) 
                      << profile.duration_ms() << " ms" << std::endl;
            std::cout << "  Memory Delta: " << profile.memory_delta_mb() << " MB" << std::endl;
            std::cout << "  Estimated FLOPs: " << profile.flops_estimate << std::endl;
            
            if (!profile.input_shapes.empty()) {
                std::cout << "  Input Shapes: ";
                for (const auto& shape : profile.input_shapes) {
                    std::cout << "[";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << shape[i];
                    }
                    std::cout << "] ";
                }
                std::cout << std::endl;
            }
            
            if (!profile.output_shapes.empty()) {
                std::cout << "  Output Shapes: ";
                for (const auto& shape : profile.output_shapes) {
                    std::cout << "[";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << shape[i];
                    }
                    std::cout << "] ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    void saveModelInfo(const std::string& output_file) {
        std::cout << "\n[SAVING MODEL INFORMATION]" << std::endl;
        
        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        out << "PyTorch Model Analysis Report\n";
        out << std::string(50, '=') << "\n\n";
        
        // Model summary
        out << "Model Parameters:\n";
        for (const auto& param_pair : model_parameters) {
            const auto& param = param_pair.second;
            out << "  " << param.name << ": shape [";
            for (size_t i = 0; i < param.tensor_info.shape.size(); ++i) {
                if (i > 0) out << ", ";
                out << param.tensor_info.shape[i];
            }
            out << "] (" << param.tensor_info.numel() << " elements)\n";
        }
        
        // Graph structure
        out << "\nGraph Structure:\n";
        for (const auto& node : nodes) {
            out << "  " << node.name << " [" << getNodeTypeName(node.type) << "]\n";
            out << "    PyTorch Op: " << node.pytorch_op << "\n";
            out << "    Parameters: " << node.total_parameters() << "\n";
            out << "    Estimated FLOPs: " << node.flops_estimate << "\n";
        }
        
        out.close();
        std::cout << "[SUCCESS] Model information saved to: " << output_file << std::endl;
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <pytorch_model.pt> [options]" << std::endl;
    std::cout << "Direct PyTorch model parser with inference and instrumentation." << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --inference     Execute inference with sample input" << std::endl;
    std::cout << "  --profile       Enable detailed profiling during inference" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name << " resnet18.pt --inference --profile" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Direct PyTorch Model Parser with Inference" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    bool run_inference = false;
    bool enable_profiling = false;
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--inference") {
            run_inference = true;
        } else if (arg == "--profile") {
            enable_profiling = true;
            run_inference = true; // Profiling requires inference
        }
    }
    
    try {
        // Create parser and load model
        DirectPyTorchParser parser(model_path);
        
        // Parse the graph structure
        parser.parseGraphStructure();
        
        // Print model summary
        parser.printModelSummary();
        
        // Run inference if requested
        if (run_inference) {
            // Create sample input (adjust dimensions as needed for your model)
            auto sample_input = torch::randn({1, 3, 224, 224}); // Common input size for vision models
            
            std::cout << "\n[CREATING SAMPLE INPUT]" << std::endl;
            std::cout << "Sample input created with shape: [1, 3, 224, 224]" << std::endl;
            std::cout << "Note: Adjust input dimensions in code for your specific model" << std::endl;
            
            // Execute inference
            auto output = parser.executeInference(sample_input, enable_profiling);
            
            // Print results
            if (enable_profiling) {
                parser.printInstrumentationResults();
            }
            
            std::cout << "\n[INFERENCE RESULTS]" << std::endl;
            std::cout << "Output tensor shape: [";
            for (int i = 0; i < output.dim(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << output.size(i);
            }
            std::cout << "]" << std::endl;
            
            // Show sample output values
            if (output.numel() <= 20) {
                std::cout << "Output values: " << output << std::endl;
            } else {
                std::cout << "First few output values: " << output.flatten().slice(0, 0, 10) << std::endl;
            }
        }
        
        // Save detailed model information
        parser.saveModelInfo("model_analysis.txt");
        
        std::cout << "\n[SUCCESS] Analysis completed!" << std::endl;
        std::cout << "Model parsed directly from .pt file without requiring .fbs schema." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}