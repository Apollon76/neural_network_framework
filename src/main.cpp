#include <iostream>
#include <armadillo>

int main() {
    std::cout << "Hello, world!" << std::endl;

    arma::mat A = arma::randu<arma::mat>(4, 5);
    arma::mat B = arma::randu<arma::mat>(4, 5);

    std::cout << A * B.t() << std::endl;

    return 0;
}
