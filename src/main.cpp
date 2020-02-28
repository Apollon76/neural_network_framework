#include <vector>
#include <armadillo>

class TensorDimensions {
public:
private:

};

class DenseLayer {
public:

private:

};

class NeuralNetwork {
public:
    NeuralNetwork() {}

private:

};

int main(int argc, char **argv) {
    std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;

    arma::mat A(2, 3);  // directly specify the matrix size (elements are uninitialised)

    std::cout << "A.n_rows: " << A.n_rows << std::endl;  // .n_rows and .n_cols are read only
    std::cout << "A.n_cols: " << A.n_cols << std::endl;

    A(1, 2) = 456.0;  // directly access an element (indexing starts at 0)
    A.print("A:");

    A = 5.0;         // scalars are treated as a 1x1 matrix
    A.print("A:");

    A.set_size(4, 5); // change the size (data is not preserved)

    A.fill(5.0);     // set all elements to a particular value
    A.print("A:");

    // endr indicates "end of row"
    A << 0.165300 << 0.454037 << 0.995795 << 0.124098 << 0.047084 << arma::endr
      << 0.688782 << 0.036549 << 0.552848 << 0.937664 << 0.866401 << arma::endr
      << 0.348740 << 0.479388 << 0.506228 << 0.145673 << 0.491547 << arma::endr
      << 0.148678 << 0.682258 << 0.571154 << 0.874724 << 0.444632 << arma::endr
      << 0.245726 << 0.595218 << 0.409327 << 0.367827 << 0.385736 << arma::endr;

    A.print("A:");

    // determinant
    std::cout << "det(A): " << arma::det(A) << std::endl;

    // inverse
    std::cout << "inv(A): " << std::endl << arma::inv(A) << std::endl;

    // save matrix as a text file
    A.save("A.txt", arma::raw_ascii);

    // load from file
    arma::mat B;
    B.load("A.txt");

    // submatrices
    std::cout << "B( span(0,2), span(3,4) ):" << std::endl << B(arma::span(0, 2), arma::span(3, 4)) << std::endl;

    std::cout << "B( 0,3, size(3,2) ):" << std::endl << B(0, 3, arma::size(3, 2)) << std::endl;

    std::cout << "B.row(0): " << std::endl << B.row(0) << std::endl;

    std::cout << "B.col(1): " << std::endl << B.col(1) << std::endl;

    // transpose
    std::cout << "B.t(): " << std::endl << B.t() << std::endl;

    // maximum from each column (traverse along rows)
    std::cout << "max(B): " << std::endl << max(B) << std::endl;

    // maximum from each row (traverse along columns)
    std::cout << "max(B,1): " << std::endl << max(B, 1) << std::endl;

    // maximum value in B
    std::cout << "max(max(B)) = " << max(max(B)) << std::endl;

    // sum of each column (traverse along rows)
    std::cout << "sum(B): " << std::endl << sum(B) << std::endl;

    // sum of each row (traverse along columns)
    std::cout << "sum(B,1) =" << std::endl << sum(B, 1) << std::endl;

    // sum of all elements
    std::cout << "accu(B): " << accu(B) << std::endl;

    // trace = sum along diagonal
    std::cout << "trace(B): " << trace(B) << std::endl;

    // generate the identity matrix
    arma::mat C = arma::eye<arma::mat>(4, 4);

    // random matrix with values uniformly distributed in the [0,1] interval
    arma::mat D = arma::randu<arma::mat>(4, 4);
    D.print("D:");

    // row vectors are treated like a matrix with one row
    arma::rowvec r;
    r << 0.59119 << 0.77321 << 0.60275 << 0.35887 << 0.51683;
    r.print("r:");

    // column vectors are treated like a matrix with one column
    arma::vec q;
    q << 0.14333 << 0.59478 << 0.14481 << 0.58558 << 0.60809;
    q.print("q:");

    // convert matrix to vector; data in matrices is stored column-by-column
    arma::vec v = vectorise(A);
    v.print("v:");

    // dot or inner product
    std::cout << "as_scalar(r*q): " << as_scalar(r * q) << std::endl;

    // outer product
    std::cout << "q*r: " << std::endl << q * r << std::endl;

    // multiply-and-accumulate operation (no temporary matrices are created)
    std::cout << "accu(A % B) = " << accu(A % B) << std::endl;

    // example of a compound operation
    B += 2.0 * A.t();
    B.print("B:");

    // imat specifies an integer matrix
    arma::imat AA;
    arma::imat BB;

    AA << 1 << 2 << 3 << arma::endr << 4 << 5 << 6 << arma::endr << 7 << 8 << 9;
    BB << 3 << 2 << 1 << arma::endr << 6 << 5 << 4 << arma::endr << 9 << 8 << 7;

    // comparison of matrices (element-wise); output of a relational operator is a umat
    arma::umat ZZ = (AA >= BB);
    ZZ.print("ZZ:");

    // cubes ("3D matrices")
    arma::cube Q(B.n_rows, B.n_cols, 2);

    Q.slice(0) = B;
    Q.slice(1) = 2.0 * B;

    Q.print("Q:");

    // 2D field of matrices; 3D fields are also supported
    arma::field<arma::mat> F(4, 3);

    for (arma::uword col = 0; col < F.n_cols; ++col)
        for (arma::uword row = 0; row < F.n_rows; ++row) {
            F(row, col) = arma::randu<arma::mat>(2, 3);  // each element in field<mat> is a matrix
        }

    F.print("F:");

    return 0;
}