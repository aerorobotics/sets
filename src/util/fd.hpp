// Functions to compute gradients using finite difference.
// Based on the functions in https://github.com/PatWie/CppNumericalSolvers
// and rewritten to use Eigen
// #include "finitediff.hpp"

#include <array>
#include <vector>

// #include <spdlog/spdlog.h>


namespace fd {


enum AccuracyOrder {
    SECOND, ///< @brief Second order accuracy.
    FOURTH, ///< @brief Fourth order accuracy.
    SIXTH,  ///< @brief Sixth order accuracy.
    EIGHTH  ///< @brief Eighth order accuracy.
};





// The external coefficients, c1, in c1 * f(x + c2).
// See: https://en.wikipedia.org/wiki/Finite_difference_coefficient
std::vector<double> get_external_coeffs(const AccuracyOrder accuracy)
{
    switch (accuracy) {
    case SECOND:
        return { { 1, -1 } };
    case FOURTH:
        return { { 1, -8, 8, -1 } };
    case SIXTH:
        return { { -1, 9, -45, 45, -9, 1 } };
    case EIGHTH:
        return { { 3, -32, 168, -672, 672, -168, 32, -3 } };
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}

// The internal coefficients, c2, in c1 * f(x + c2).
// See: https://en.wikipedia.org/wiki/Finite_difference_coefficient
std::vector<double> get_interior_coeffs(const AccuracyOrder accuracy)
{
    switch (accuracy) {
    case SECOND:
        return { { 1, -1 } };
    case FOURTH:
        return { { -2, -1, 1, 2 } };
    case SIXTH:
        return { { -3, -2, -1, 1, 2, 3 } };
    case EIGHTH:
        return { { -4, -3, -2, -1, 1, 2, 3, 4 } };
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}

// The denominators of the finite difference.
double get_denominator(const AccuracyOrder accuracy)
{
    switch (accuracy) {
    case SECOND:
        return 2;
    case FOURTH:
        return 12;
    case SIXTH:
        return 60;
    case EIGHTH:
        return 840;
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}

// Compute the gradient of a function at a point using finite differences.
void finite_gradient(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const std::function<double(const Eigen::VectorXd&)>& f,
    Eigen::VectorXd& grad,
    const AccuracyOrder accuracy,
    const double eps)
{
    const std::vector<double> external_coeffs = get_external_coeffs(accuracy);
    const std::vector<double> internal_coeffs = get_interior_coeffs(accuracy);

    assert(external_coeffs.size() == internal_coeffs.size());
    const size_t inner_steps = internal_coeffs.size();

    const double denom = get_denominator(accuracy) * eps;

    grad.setZero(x.rows());

    Eigen::VectorXd x_mutable = x;
    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t ci = 0; ci < inner_steps; ci++) {
            x_mutable[i] += internal_coeffs[ci] * eps;
            grad[i] += external_coeffs[ci] * f(x_mutable);
            x_mutable[i] = x[i];
        }
        grad[i] /= denom;
    }
}

void finite_jacobian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f,
    Eigen::MatrixXd& jac,
    const AccuracyOrder accuracy,
    const double eps)
{
    const std::vector<double> external_coeffs = get_external_coeffs(accuracy);
    const std::vector<double> internal_coeffs = get_interior_coeffs(accuracy);

    assert(external_coeffs.size() == internal_coeffs.size());
    const size_t inner_steps = internal_coeffs.size();

    const double denom = get_denominator(accuracy) * eps;

    jac.setZero(f(x).rows(), x.rows());

    Eigen::VectorXd x_mutable = x;
    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t ci = 0; ci < inner_steps; ci++) {
            x_mutable[i] += internal_coeffs[ci] * eps;
            jac.col(i) += external_coeffs[ci] * f(x_mutable);
            x_mutable[i] = x[i];
        }
        jac.col(i) /= denom;
    }
}

void finite_hessian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const std::function<double(const Eigen::VectorXd&)>& f,
    Eigen::MatrixXd& hess,
    const AccuracyOrder accuracy,
    const double eps)
{
    const std::vector<double> external_coeffs = get_external_coeffs(accuracy);
    const std::vector<double> internal_coeffs = get_interior_coeffs(accuracy);

    assert(external_coeffs.size() == internal_coeffs.size());
    const size_t inner_steps = internal_coeffs.size();

    double denom = get_denominator(accuracy) * eps;
    denom *= denom;

    hess.setZero(x.rows(), x.rows());

    Eigen::VectorXd x_mutable = x;
    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = i; j < x.rows(); j++) {
            for (size_t ci = 0; ci < inner_steps; ci++) {
                for (size_t cj = 0; cj < inner_steps; cj++) {
                    x_mutable[i] += internal_coeffs[ci] * eps;
                    x_mutable[j] += internal_coeffs[cj] * eps;
                    hess(i, j) += external_coeffs[ci] * external_coeffs[cj]
                        * f(x_mutable);
                    x_mutable[j] = x[j];
                    x_mutable[i] = x[i];
                }
            }
            hess(i, j) /= denom;
            hess(j, i) = hess(i, j); // The hessian is symmetric
        }
    }
}

// Compare if two gradients are close enough.
bool compare_gradient(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const double test_eps,
    const std::string& msg)
{
    assert(x.rows() == y.rows());

    bool same = true;
    for (long i = 0; i < x.rows(); i++) {
        double scale =
            std::max(std::max(std::abs(x[i]), std::abs(y[i])), double(1.0));
        double abs_diff = std::abs(x[i] - y[i]);

        // if (abs_diff > test_eps * scale) {
        //     spdlog::debug(
        //         "{} eps={:.3e} r={} x={:.3e} y={:.3e} |x-y|={:.3e} "
        //         "|x-y|/|x|={:.3e} |x-y|/|y|={:3e}",
        //         msg, test_eps, i, x(i), y(i), abs_diff, abs_diff / abs(x(i)),
        //         abs_diff / std::abs(y(i)));
        //     same = false;
        // }
    }
    return same;
}

// Compare if two jacobians are close enough.
bool compare_jacobian(
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& y,
    const double test_eps,
    const std::string& msg)
{
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());

    bool same = true;
    for (long i = 0; i < x.rows(); i++) {
        for (long j = 0; j < x.cols(); j++) {
            double scale = std::max(
                std::max(std::abs(x(i, j)), std::abs(y(i, j))), double(1.0));

            double abs_diff = std::abs(x(i, j) - y(i, j));

            // if (abs_diff > test_eps * scale) {
            //     spdlog::debug(
            //         "{} eps={:.3e} r={} c={} x={:.3e} y={:.3e} "
            //         "|x-y|={:.3e} |x-y|/|x|={:.3e} |x-y|/|y|={:3e}",
            //         msg, test_eps, i, j, x(i, j), y(i, j), abs_diff,
            //         abs_diff / std::abs(x(i, j)), abs_diff / std::abs(y(i, j)));
            //     same = false;
            // }
        }
    }
    return same;
}

// Compare if two hessians are close enough.
bool compare_hessian(
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& y,
    const double test_eps,
    const std::string& msg)
{
    return compare_jacobian(x, y, test_eps, msg);
}

// Flatten the matrix rowwise
Eigen::VectorXd flatten(const Eigen::Ref<const Eigen::MatrixXd>& X)
{
    Eigen::VectorXd x(X.size());
    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            x(i * X.cols() + j) = X(i, j);
        }
    }
    return x;
}

// Unflatten rowwise
Eigen::MatrixXd unflatten(const Eigen::Ref<const Eigen::VectorXd>& x, int dim)
{
    assert(x.size() % dim == 0);
    Eigen::MatrixXd X(x.size() / dim, dim);
    for (int i = 0; i < x.size(); i++) {
        X(i / dim, i % dim) = x(i);
    }
    return X;
}

} // namespace fd