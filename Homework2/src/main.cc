#include <iostream>

#include "algorithm.h"
#include "matrix.h"

void question_1() {
    std::cout << "1" << std::endl;
    double eps = 1e-6;
    matrix x{{0}, {0}};
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return 2 * sqr(x_1) + sqr(x_2) + 2 * x_1 * x_2 + x_1 - x_2;
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{{4 * x_1 + 2 * x_2 + 1}, {2 * x_2 + 2 * x_1 - 1}};
    };
    matrix result = conjugate_gradient(f, df, x, eps);
    std::cout << result << std::endl;
    std::cout << "Optimized value = " << f(result) << std::endl;
    std::cout << std::endl;
}

void question_2() {
    std::cout << "2" << std::endl;
    auto f = [](double x) -> double { return 2 * x * x - x - 1; };
    auto g = [](double x) -> double { return 3 * x * x - 21.6 * x - 1; };
    auto df = [](double x) -> double { return 4 * x - 1; };
    auto dg = [](double x) -> double { return 6 * x - 21.6; };

    std::cout
        << "golden ratio\n"
        << golden_ratio(f, -1, 1, 0.06)
        << '\t'
        << golden_ratio(g, 0, 25, 0.08)
        << std::endl;
    std::cout << "fibonacci\n"
              << fibonacci(f, -1, 1, 0.06)
              << '\t'
              << fibonacci(g, 0, 25, 0.08)
              << std::endl;
    std::cout << "binary search\n"
              << binary_search(df, -1, 1, 0.06)
              << '\t'
              << binary_search(dg, 0, 25, 0.08)
              << std::endl;
    std::cout << "dichotomous\n"
              << dichotomous(f, -1, 1, 0.06)
              << '\t'
              << dichotomous(g, 0, 25, 0.08)
              << std::endl;
    std::cout << std::endl;
}

void question_3() {
    std::cout << "3" << std::endl;
    matrix x{
        {-1},
        {1}};
    matrix d{
        {1},
        {1}};
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return 100 * sqr(x_2 - sqr(x_1)) + sqr(1 - x_1);
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{
            {400 * x_1 * (sqr(x_1) - x_2) + 2 * (x_1 - 1)},
            {200 * (x_2 - sqr(x_1))}};
    };
    std::cout << "Goldstein\n"
              << goldstein(f, df, x, d) << std::endl;
    std::cout << "Wolfe-Powell\n"
              << wolfe_powell(f, df, x, d) << std::endl;
    std::cout << std::endl;
}

void question_5() {
    std::cout << "5" << std::endl;
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return 10 * sqr(x_1) + sqr(x_2);
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{{20 * x_1}, {2 * x_2}};
    };
    matrix x{{0.1}, {1}};
    matrix H = matrix::I(2);
    std::cout << "DFP" << std::endl;
    matrix result = dfp(f, df, H, x);
    std::cout << result << std::endl;
    std::cout << "Optimized value = " << f(result) << std::endl;
    std::cout << std::endl;
}

void question_6() {
    std::cout << "6" << std::endl;
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return sqr(x_1) + 4 * sqr(x_2) - 4 * x_1 - 8 * x_2;
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{{2 * x_1 - 4}, {8 * x_2 - 8}};
    };
    matrix x{{0}, {0}};
    matrix H = matrix::I(2);
    std::cout << "BFGS" << std::endl;
    matrix result = dfp(f, df, H, x);
    std::cout << result << std::endl;
    std::cout << "Optimized value = " << f(result) << std::endl;
    std::cout << std::endl;
}

void question_7() {
    std::cout << "7" << std::endl;
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return sqr(x_1) + sqr(x_2) - x_1 * x_2 - 10 * x_1 - 4 * x_2 + 60;
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{{2 * x_1 - x_2 - 10}, {2 * x_2 - x_1 - 4}};
    };
    matrix x{{0}, {0}};
    matrix H = matrix::I(2);
    std::cout << "DFP" << std::endl;
    matrix dfp_result = dfp(f, df, H, x);
    std::cout << dfp_result << std::endl;
    std::cout << "Optimized value = " << f(dfp_result) << std::endl;
    std::cout << "BFGS" << std::endl;
    matrix bfgs_result = dfp(f, df, H, x);
    std::cout << bfgs_result << std::endl;
    std::cout << "Optimized value = " << f(bfgs_result) << std::endl;
    std::cout << "FR conjugate gradient" << std::endl;
    matrix result = conjugate_gradient(f, df, x, 1e-6);
    std::cout << result << std::endl;
    std::cout << "Optimized value = " << f(result) << std::endl;
    std::cout << std::endl;
}

void question_10() {
    std::cout << "10" << std::endl;
    auto f = [](const matrix &a) -> double {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return sqr(x_1) + sqr(x_2) - x_1 * x_2 - 10 * x_1 - 4 * x_2 + 60;
    };
    auto df = [](const matrix &a) -> matrix {
        assert(a.line() == 2 && a.column() == 1);
        double x_1 = a[0][0], x_2 = a[1][0];
        return matrix{{2 * x_1 - x_2 - 10}, {2 * x_2 - x_1 - 4}};
    };
    matrix x{{0}, {0}};
    std::cout << "SGD" << std::endl;
    matrix result = sgd(f, df, x);
    std::cout << result << std::endl;
    std::cout << "Optimized value = " << f(result) << std::endl;
}

int main() {
    std::cout.setf(std::ios::fixed);
    question_1();
    question_2();
    question_3();
    question_5();
    question_6();
    question_7();
    question_10();
    return 0;
}
