#include <torch/script.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " model.pt [cifar_batch_file image_index]\n";
        return 1;
    }

    try {
        torch::NoGradGuard no_grad;
        torch::jit::ExtraFilesMap extra_files{{"classes.txt", ""}};
        auto model = torch::jit::load(argv[1], torch::kCPU, extra_files);
        model.eval();

        std::vector<std::string> classes;
        std::istringstream class_names(extra_files.at("classes.txt"));
        std::string name;
        while (std::getline(class_names, name)) {
            classes.push_back(name);
        }
        if (classes.size() != 10) {
            throw std::runtime_error("Expected the 10 class names saved by export_model.py.");
        }

        auto input = torch::ones({1, 3, 32, 32}, torch::kFloat32);
        int expected_label = -1;
        if (argc == 4) {
            std::size_t parsed = 0;
            const std::string index_text = argv[3];
            const int index = std::stoi(index_text, &parsed);
            if (parsed != index_text.size() || index < 0 || index >= 10000) {
                throw std::invalid_argument("Image index must be between 0 and 9999.");
            }

            // CIFAR's binary format stores one label followed by planar RGB bytes.
            std::ifstream batch(argv[2], std::ios::binary);
            if (!batch) {
                throw std::runtime_error("Cannot open the CIFAR binary batch file.");
            }
            std::array<unsigned char, 1 + 3 * 32 * 32> record{};
            batch.seekg(static_cast<std::streamoff>(index) * record.size());
            batch.read(reinterpret_cast<char*>(record.data()), record.size());
            if (batch.gcount() != static_cast<std::streamsize>(record.size())) {
                throw std::runtime_error("The requested image is missing or incomplete.");
            }
            expected_label = record[0];
            if (expected_label >= 10) {
                throw std::runtime_error("Invalid CIFAR class label.");
            }

            input = torch::from_blob(record.data() + 1, {1, 3, 32, 32}, torch::kUInt8)
                        .to(torch::kFloat32);
            input = (input / 255.0 - 0.5) / 0.5;
        } else {
            std::cout << "Input: synthetic tensor of ones (pipeline check only)\n";
        }

        auto logits = model.forward({input}).toTensor();
        const int prediction = logits.argmax(1).item<int>();
        std::cout << "Logits:" << std::setprecision(9);
        for (int i = 0; i < 10; ++i) {
            std::cout << ' ' << logits[0][i].item<float>();
        }
        std::cout << "\nPredicted class: " << classes.at(prediction)
                  << " (" << prediction << ")\n";
        if (expected_label >= 0) {
            std::cout << "Actual class: " << classes.at(expected_label) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
