#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

    return EXIT_SUCCESS;
}
