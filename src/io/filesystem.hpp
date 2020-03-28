#pragma once

#include <experimental/filesystem>
#include <vector>
#include <algorithm>

namespace nn_framework::io {
    class Filesystem {
    public:
        static std::vector<std::string> ListFiles(const std::string& directory) {
            std::vector<std::string> paths;
            for (const auto& path : std::experimental::filesystem::directory_iterator(directory)) {
                paths.emplace_back(path.path().string());
            }
            std::sort(paths.begin(), paths.end());
            return paths;
        }

        static std::string GetFilename(const std::string& path) {
            return std::experimental::filesystem::path(path).filename();
        }
    };
}