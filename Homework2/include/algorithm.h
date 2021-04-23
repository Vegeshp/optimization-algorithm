#pragma once

#ifndef ALGORITHM_H_
#define ALGORITHM_H_

#include <functional>
#include <random>

#include "matrix.h"

double sqr(double x);

matrix gradient(const matrix &A, const matrix &B, const matrix &x);

matrix conjugate_gradient(const std::function<double(matrix)> &f,
                          const std::function<matrix(matrix)> &df,
                          matrix x, double eps);

double golden_ratio(const std::function<double(double)> &f,
                    double alpha, double beta, double eps);

double fibonacci(const std::function<double(double)> &f,
                 double alpha, double beta, double eps);

double binary_search(const std::function<double(double)> &f,
                     double alpha, double beta, double eps);

double dichotomous(const std::function<double(double)> &f,
                   double alpha, double beta, double eps);

double goldstein(const std::function<double(const matrix &)> &f,
                 const std::function<matrix(const matrix &)> &df,
                 matrix x, const matrix &d);

double wolfe_powell(const std::function<double(const matrix &)> &f,
                    const std::function<matrix(const matrix &)> &df,
                    matrix x, const matrix &d);

double dfp(const std::function<double(const matrix &)> &f,
           const std::function<matrix(const matrix &)> &df,
           matrix H, matrix x);

double bfgs(const std::function<double(const matrix &)> &f,
            const std::function<matrix(const matrix &)> &df,
            matrix H, matrix x);

// Implementation

double sqr(double x) {
    return x * x;
}

double arg_min(const std::function<double(matrix)> &f, const matrix &x, const matrix &grad) {
    static std::mt19937_64 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dis{0, 1};
    double alpha = 0;
    double min_value = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 100; i++) {
        double alpha_temp = dis(gen);
        double temp = f(x + (alpha_temp * grad));
        if (min_value > temp) {
            min_value = temp;
            alpha = alpha_temp;
        }
    }
    return alpha;
}

matrix conjugate_gradient(const std::function<double(matrix)> &f,
                          const std::function<matrix(matrix)> &df,
                          matrix x, double eps) {
    matrix delta_x = -df(x);
    matrix s = delta_x;
    x = x + arg_min(f, x, delta_x) * delta_x;
    while(delta_x.norm() > eps) {
        matrix new_delta_x = -df(x);
        double beta = ((new_delta_x.T() * new_delta_x) / (delta_x.T() * delta_x))[0][0];
        s = new_delta_x + beta * s;
        x = x + (arg_min(f, x, s) * s);
        delta_x = new_delta_x;
    }
    return x;
}

double golden_ratio(const std::function<double(double)> &f,
                    double alpha, double beta, double eps) {
    static const double ratio = (std::sqrt(5) - 1) / 2;
    double lambda = alpha + (1 - ratio) * (beta - alpha);
    double mu = alpha + ratio * (beta - alpha);
    while (beta - alpha >= eps) {
        if (f(lambda) > f(mu)) {
            alpha = lambda;
            lambda = mu;
            mu = alpha + ratio * (beta - alpha);
        } else {
            beta = mu;
            mu = lambda;
            lambda = alpha + (1 - ratio) * (beta - alpha);
        }
    }
    return (alpha + beta) / 2;
}

double fibonacci(const std::function<double(double)> &f,
                 double alpha, double beta, double eps) {
    using ll = int64_t;
    // ratio = f1 / f2
    ll fn = 13, fn1 = 8;
    double ratio = 1.0 * fn1 / fn;
    double lambda = alpha + (1 - ratio) * (beta - alpha);
    double mu = alpha + ratio * (beta - alpha);
    while (beta - alpha >= eps) {
        ll temp = fn;
        fn += fn1;
        fn1 = temp;
        ratio = 1.0 * fn1 / fn;
        if (f(lambda) > f(mu)) {
            alpha = lambda;
            lambda = mu;
            mu = alpha + ratio * (beta - alpha);
        } else {
            beta = mu;
            mu = lambda;
            lambda = alpha + (1 - ratio) * (beta - alpha);
        }
    }
    return (alpha + beta) / 2;
}

double binary_search(const std::function<double(double)> &f,
                     double alpha, double beta, double eps) {
    while (beta - alpha >= eps) {
        double mid = (alpha + beta) / 2;
        double para = f(mid);
        if (std::fabs(para) < eps) {
            return mid;
        } else if (para > 0) {
            beta = mid;
        } else if (para < 0) {
            alpha = mid;
        }
    }
    return (alpha + beta) / 2;
}

double dichotomous(const std::function<double(double)> &f,
                   double alpha, double beta, double eps) {
    while (beta - alpha > (2 + eps) * eps) {
        double lambda = (alpha + beta) / 2 - eps;
        double mu = (alpha + beta) / 2 + eps;
        if (f(lambda) < f(mu)) {
            beta = mu;
        } else {
            alpha = lambda;
        }
    }
    return (alpha + beta) / 2;
}

double goldstein(const std::function<double(const matrix &)> &f,
                 const std::function<matrix(const matrix &)> &df,
                 matrix x, const matrix &d) {
    const double rho = 0.1;
    double rhs = (df(x).T() * d)[0][0];
    double value = f(x);
    static std::mt19937_64 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dis{0, 1};
    while (true) {
        bool flag = false;
        double lambda, new_value;
        for (int i = 0; i < 100; i++) {
            lambda = dis(gen);
            new_value = f(x + (lambda * d));
            double ratio = (new_value - value) / (rhs * lambda);
            if (ratio > rho && ratio < 1) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            break;
        }
        x = x + (lambda * d);
        value = new_value;
        rhs = (df(x).T() * d)[0][0];
    }
    return value;
}

double wolfe_powell(const std::function<double(const matrix &)> &f,
                    const std::function<matrix(const matrix &)> &df,
                    matrix x, const matrix &d) {
    const double rho = 0.1;
    double rhs = (df(x).T() * d)[0][0];
    double value = f(x);
    static std::mt19937_64 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dis{0, 1};
    while (true) {
        bool flag = false;
        double lambda, new_value;
        for (int i = 0; i < 100; i++) {
            lambda = dis(gen);
            new_value = f(x + (lambda * d));
            double ratio = -std::abs((df(x + (lambda * d)).T() * d)[0][0]) / rhs;
            if (new_value - value > rhs * lambda * rho &&
                ratio > 0 && ratio < 1) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            break;
        }
        x = x + (lambda * d);
        value = new_value;
        rhs = (df(x).T() * d)[0][0];
    }
    return value;
}

/**
 * 4.
 * subdifferential
 * a) (x, y, z), x, y, z \in [-1, 1]
 * b) [-1, 1]
 * c) grad = (1, x), x \in [-1, 1]
 */

double dfp(const std::function<double(const matrix &)> &f,
           const std::function<matrix(const matrix &)> &df,
           matrix H, matrix x) {
    const double eps = 1e-4;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis{0, 1};
    matrix grad = df(x);
    while (grad.norm() > eps) {
        matrix d = -H * grad;
        double ans = std::numeric_limits<double>::infinity(), alpha = 0;
        for (int i = 0; i < 1000; i++) {
            double t = dis(gen);
            double temp = f(x + t * d);
            if (ans > temp) {
                ans = temp;
                alpha = t;
            }
        }
        matrix delta_x = alpha * d;
        x = x + delta_x;
        matrix delta_grad = df(x) - grad;
        matrix numerator = H * grad;
        H = H + ((delta_x * delta_x.T()) / (delta_x.T() * delta_grad)) - (((numerator * numerator.T())) / (delta_grad.T() * H * delta_grad));
        grad = grad + delta_grad;
    }
    return f(x);
}

double bfgs(const std::function<double(const matrix &)> &f,
            const std::function<matrix(const matrix &)> &df,
            matrix H, matrix x) {
    const double eps = 1e-8;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis{0, 1};
    matrix grad = df(x);
    while (grad.norm() > eps) {
        matrix d = -H * grad;
        double ans = std::numeric_limits<double>::infinity(), alpha = 0;
        for (int i = 0; i < 100; i++) {
            double t = dis(gen);
            double temp = f(x + t * d);
            if (ans > temp) {
                ans = temp;
                alpha = t;
            }
        }
        matrix delta_x = alpha * d;
        x = x + delta_x;
        matrix delta_grad = df(x) - grad;
        matrix temp = H * delta_grad * delta_x.T();
        H = H + (1 + ((delta_grad.T() * H * delta_grad) / (delta_grad.T() * delta_x))[0][0]) * ((delta_x * delta_x.T()) / (delta_x.T() * delta_grad)) -
            (temp + temp.T()) / (delta_grad.T() * delta_x);
        grad = grad + delta_grad;
    }
    return f(x);
}

#endif  // ALGORITHM_H_
