#pragma once

#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png

#include <utility>
#include <vector>
#include <armadillo>
#include <CImg.h>
#include <src/utils.hpp>
#include "filesystem.hpp"

namespace nn_framework::io {
    class ImgReader {
    public:
        explicit ImgReader(std::vector<std::string> input_paths) :
            input_paths(std::move(input_paths)) {
        }

        ImgReader(std::initializer_list<std::string> input_paths) :
            input_paths(input_paths) {
        }

        template<typename T>
        std::vector<arma::Cube<T>> LoadData() {
            std::vector<arma::Cube<T>> result;
            for (const auto &path : input_paths) {
                result.emplace_back(ImgToCube(cimg_library::CImg<T>(path.c_str())));
            }
            return result;
        }

        template<typename T>
        std::vector<std::tuple<std::string, arma::Cube<T>>> LoadDataWithNames() {
            std::vector<std::tuple<std::string, arma::Cube<T>>> result;
            for (const auto &path : input_paths) {
                result.emplace_back(Filesystem::GetFilename(path), ImgToCube(cimg_library::CImg<T>(path.c_str())));
            }
            return result;
        }

        template<typename T>
        arma::Cube<T> ImgToCube(const cimg_library::CImg<T>& image) {
            ensure(image.depth() == 1, "only 2d images are supported");
            auto cube = arma::Cube<T>(image.height(), image.width(), image.spectrum());
            for (auto i = 0; i < image.height(); i++) {
                for (auto j = 0; j < image.width(); j++) {
                    for (auto k = 0; k < image.spectrum(); k++) {
                        cube.at(i, j, k) = image(i, j, 0, k);
                    }
                }
            }
            return cube;
        }

    private:
        std::vector<std::string> input_paths;
    };

    class ImgWriter {
    public:
        template<typename T>
        void WritePng(const arma::Cube<T>& img, const std::string& path) {
            cimg_library::CImg<T> image(img.n_rows, img.n_cols, 1, img.n_slices);
            for (auto i = 0; i < image.height(); i++) {
                for (auto j = 0; j < image.width(); j++) {
                    for (auto k = 0; k < image.spectrum(); k++) {
                        image(i, j, 0, k) = img.at(i, j, k);
                    }
                }
            }
            image.save_png(path);
        }
    };
}