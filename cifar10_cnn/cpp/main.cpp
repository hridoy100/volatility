#include <torch/script.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestImages {
    torch::Tensor pixels;
    torch::Tensor labels;
};

TestImages read_test_images(const std::string& path) {
    constexpr std::int64_t record_size = 1 + 3 * 32 * 32;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open test images: " + path);
    }
    const std::streamoff byte_count = file.tellg();
    if (byte_count <= 0 || byte_count % record_size != 0) {
        throw std::runtime_error("Test file must contain complete 3073-byte CIFAR records.");
    }

    std::vector<unsigned char> bytes(static_cast<std::size_t>(byte_count));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), byte_count);
    if (file.gcount() != byte_count) {
        throw std::runtime_error("Could not read the complete test file.");
    }

    const std::int64_t count = byte_count / record_size;
    // Clone so the tensors own their storage after the byte vector goes away.
    auto records = torch::from_blob(bytes.data(), {count, record_size}, torch::kUInt8).clone();
    auto labels = records.select(1, 0).to(torch::kInt64);
    if (labels.max().item<int>() >= 10) {
        throw std::runtime_error("Test file contains an invalid CIFAR class label.");
    }
    auto pixels = records.slice(1, 1).reshape({count, 3, 32, 32});
    return {pixels, labels};
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " model.pt test_images.bin predictions.csv\n";
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

        const auto test = read_test_images(argv[2]);
        const auto output_path = std::filesystem::weakly_canonical(argv[3]);
        if (output_path == std::filesystem::canonical(argv[1]) ||
            output_path == std::filesystem::canonical(argv[2])) {
            throw std::invalid_argument("Predictions must not overwrite the model or test images.");
        }
        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("Cannot open the predictions CSV for writing.");
        }
        output << "image_index,actual_label,predicted_label,actual_class,predicted_class,confidence,correct\n";
        output << std::fixed << std::setprecision(8);

        constexpr std::int64_t batch_size = 128;
        const std::int64_t total = test.labels.size(0);
        std::int64_t correct = 0;
        for (std::int64_t start = 0; start < total; start += batch_size) {
            const auto end = std::min(start + batch_size, total);
            auto images = test.pixels.slice(0, start, end).to(torch::kFloat32);
            images = (images / 255.0 - 0.5) / 0.5;
            auto logits = model.forward({images}).toTensor();
            auto predictions = logits.argmax(1);
            auto confidence = logits.softmax(1).gather(1, predictions.unsqueeze(1)).squeeze(1);

            for (std::int64_t row = 0; row < end - start; ++row) {
                const auto index = start + row;
                const int actual = test.labels[index].item<int>();
                const int predicted = predictions[row].item<int>();
                const bool matches = actual == predicted;
                correct += matches;
                output << index << ',' << actual << ',' << predicted << ','
                       << classes.at(actual) << ',' << classes.at(predicted) << ','
                       << confidence[row].item<float>() << ',' << matches << '\n';
                if (index < 10) {
                    std::cout << "Image " << index << ": predicted=" << classes.at(predicted)
                              << ", actual=" << classes.at(actual) << '\n';
                }
            }
        }

        output.flush();
        if (!output) {
            throw std::runtime_error("Writing the predictions CSV failed.");
        }
        std::cout << "Test images: " << total << "\nCorrect: " << correct << '/' << total
                  << "\nAccuracy: " << std::fixed << std::setprecision(2)
                  << 100.0 * correct / total << "%\nPredictions: " << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
