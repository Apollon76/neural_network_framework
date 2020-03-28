#pragma once

#include <utility>
#include <vector>
#include <csv/csv.h>
#include <src/utils.hpp>

namespace nn_framework::io {
    class CsvReader {
     public:
        explicit CsvReader(const std::string& input_path, bool skip_header=false) :
            reader(input_path),
            skip_header(skip_header) {
        }

        template<typename T>
        std::vector<std::vector<T>> LoadData() {
            if (skip_header) {
                reader.next_line();
            }
            std::vector<std::vector<T>> result;
            while (auto raw_line = reader.next_line()) {
                result.push_back(ParseTokens<T>(raw_line, ','));
            }
            return result;
        }

     private:
        template<typename T>
        static std::vector<T> ParseTokens(const std::string& line, char delim) {
            std::vector<T> result;
            std::string token;
            for (char ch : line) {
                if (ch == delim) {
                    result.push_back(ParseToken<T>(token));
                    token.clear();
                } else {
                    token += ch;
                }
            }
            result.push_back(ParseToken<T>(token));
            return result;
        }

        template<typename T>
        static typename std::enable_if<std::is_same<T, int>::value, int>::type ParseToken(const std::string& token) {
            return std::stoi(token);
        }

        template<typename T>
        static typename std::enable_if<std::is_same<T, double>::value, double>::type ParseToken(const std::string& token) {
            return std::stod(token);
        }

        ::io::LineReader reader;
        bool skip_header;
    };

    class CsvWriter {
    public:
        explicit CsvWriter(const std::string& input_path) : stream(input_path) {
        }

        template<typename T>
        void WriteRow(const std::initializer_list<T>& list) {
            bool first = true;
            for (const auto& item : list) {
                if (!first) {
                    WriteDelim();
                }
                WriteToken(item);
                first = false;
            }
            WriteLineBreak();
        }

    private:
        template<typename T>
        void WriteToken(const T& token) {
            stream << token;
        }

        void WriteDelim() {
            WriteToken(',');
        }

        void WriteLineBreak() {
            WriteToken('\n');
        }

        std::ofstream stream;
    };
}