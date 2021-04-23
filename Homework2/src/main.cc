#include <iostream>

#include "algorithm.h"
#include "matrix.h"

void question_1() {
    double eps = 1e-6;
    matrix A{{4, 2},
             {2, 2}},
        B{{1},
          {-1}},  // 2 * 1
        x{{0},
          {0}};
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
    std::cout << (0.5 * result.T() * A * result) + (B.T() * result) << std::endl;
    std::cout << std::endl;
}

void question_2() {
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
    std::cout << dfp(f, df, H, x) << std::endl;
    std::cout << std::endl;
}

void question_6() {
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
    std::cout << bfgs(f, df, H, x) << std::endl;
    std::cout << std::endl;
}

void question_7() {
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
    std::cout << dfp(f, df, H, x) << std::endl;
    std::cout << "BFGS" << std::endl;
    std::cout << bfgs(f, df, H, x) << std::endl;
    std::cout << "FR conjugate gradient" << std::endl;
    matrix result = conjugate_gradient(f, df, x, 1e-6);
    std::cout << f(result) << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout.setf(std::ios::fixed);
    question_1();
    // question_2();
    // question_3();
    // question_5();
    // question_6();
    question_7();
    return 0;
}
