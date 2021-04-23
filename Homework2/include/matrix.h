#pragma once

#ifndef MATRIX_H_
#define MATRIX_H_

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <vector>

struct matrix {
   public:
    matrix(size_t line, size_t column);

    matrix(const std::initializer_list<std::initializer_list<double>> &data);

    matrix(const std::vector<std::vector<double>> &data);

    matrix(std::vector<std::vector<double>> &&data);

    size_t line() const;

    size_t column() const;

    std::vector<double> &operator[](size_t index);

    const std::vector<double> &operator[](size_t index) const;

    double norm() const;

    matrix operator+(const matrix &other) const;

    matrix operator-(const matrix &other) const;

    matrix operator-() const;

    matrix operator*(const matrix &other) const;

    matrix operator*(double x) const;

    matrix operator/(double x) const;

    matrix operator/(const matrix &other) const;

    matrix T() const;

    matrix inv() const;

    static matrix I(size_t n);

   private:
    size_t line_, column_;
    std::vector<std::vector<double>> data_;
};

std::ostream &operator<<(std::ostream &os, const matrix &v);

// Implementation

matrix::matrix(size_t line, size_t column) : line_{line},
                                             column_{column},
                                             data_(line_, std::vector<double>(column_)) {}

matrix::matrix(const std::initializer_list<std::initializer_list<double>> &data) {
    line_ = data.size();
    assert(line_ > 0);
    column_ = data.begin()->size();
    for (const std::initializer_list<double> &list : data) {
        assert(list.size() == column_);
    }
    data_.resize(line_);
    std::initializer_list<std::initializer_list<double>>::const_iterator begin = data.begin();
    size_t index = 0;
    while (begin != data.end()) {
        data_[index++] = *begin++;
    }
}

matrix::matrix(const std::vector<std::vector<double>> &data) {
    size_t l = data.size();
    for (size_t i = 0; i < l; i++) {
        assert(data[i].size() == l);
    }
    data_ = data;
}

matrix::matrix(std::vector<std::vector<double>> &&data) {
    size_t l = data.size();
    for (size_t i = 0; i < data.size(); i++) {
        assert(data[i].size() == l);
    }
    data_.swap(data);
}

size_t matrix::line() const { return line_; }

size_t matrix::column() const { return column_; }

std::vector<double> &matrix::operator[](size_t index) {
    return data_[index];
}

const std::vector<double> &matrix::operator[](size_t index) const {
    return data_[index];
}

double matrix::norm() const {
    double sum = 0;
    for (size_t i = 0; i < line_; i++) {
        for (size_t j = 0; j < column_; j++) {
            sum += data_[i][j] * data_[i][j];
        }
    }
    return std::sqrt(sum);
}

matrix matrix::operator+(const matrix &other) const {
    size_t n = line(), m = column();
    assert(n == other.line() && m == other.column());
    matrix result(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] = (*this)[i][j] + other[i][j];
        }
    }
    return result;
}

matrix matrix::operator-(const matrix &other) const {
    size_t n = line(), m = column();
    assert(n == other.line() && m == other.column());
    matrix result(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] = (*this)[i][j] - other[i][j];
        }
    }
    return result;
}

matrix matrix::operator-() const {
    matrix result = *this;
    size_t n = line(), m = column();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] = -result[i][j];
        }
    }
    return result;
}

matrix matrix::operator*(const matrix &other) const {
    size_t n = line(), m = other.column(), length = column();
    matrix result(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] = 0;
            for (size_t k = 0; k < length; k++) {
                result[i][j] += (*this)[i][k] * other[k][j];
            }
        }
    }
    return result;
}

matrix matrix::operator*(double x) const {
    matrix result = *this;
    size_t n = line(), m = column();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] *= x;
        }
    }
    return result;
}

matrix matrix::operator/(double x) const {
    matrix result = *this;
    size_t n = line(), m = column();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] /= x;
        }
    }
    return result;
}

matrix matrix::operator/(const matrix &other) const {
    if (other.line() != 1 || other.column() != 1) {
        std::cout << other.line() << ' ' << other.column() << std::endl;
        std::cout << line() << ' ' << column() << std::endl;
        assert(other.line() == 1 && other.column() == 1);
    }
    matrix result = *this;
    double key = other[0][0];
    size_t n = line(), m = column();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[i][j] /= key;
        }
    }
    return result;
}

matrix matrix::I(size_t n) {
    matrix result(n, n);
    for (size_t i = 0; i < n; i++) {
        result[i][i] = 1;
    }
    return result;
}

matrix matrix::T() const {
    size_t n = line(), m = column();
    matrix result{m, n};
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[j][i] = (*this)[i][j];
        }
    }
    return result;
}

matrix matrix::inv() const {
    assert(line() == column());
    size_t n = line();
    matrix result = *this, E = matrix::I(n);
    for (size_t i = 0; i < n; i++) {
        size_t index = i;
        while (index < n && result[i][i] == 0) {
            std::swap(result[i], result[index]);
            std::swap(E[i], E[index]);
            ++index;
        }
        if (index == n) {
            std::cout << "Not Invertible 1" << std::endl;
            assert(false);
        }
        double ratio = result[i][i];
        for (size_t j = 0; j < n; j++) {
            result[i][j] /= ratio;
            E[i][j] /= ratio;
        }
        for (size_t k = i + 1; k < n; k++) {
            if (result[k][i] == 0) {
                continue;
            }
            double ratio = -result[k][i];
            for (size_t j = 0; j < n; j++) {
                result[k][j] += ratio * result[i][j];
                E[k][j] += ratio * E[i][j];
            }
        }
    }
    for (size_t i = n - 1; i < n; i--) {
        if (result[i][i] == 0) {
            std::cout << "Not Invertible 2" << std::endl;
            assert(false);
        }
        for (size_t j = i - 1; j < n; j--) {
            double ratio = result[j][i];
            for (size_t k = 0; k < n; k++) {
                E[j][k] -= ratio * E[i][k];
            }
            result[j][i] = 0;
        }
    }
    return E;
}

matrix operator*(double x, const matrix &mat) {
    return mat * x;
}

std::ostream &operator<<(std::ostream &os, const matrix &v) {
    for (size_t i = 0; i < v.line(); i++) {
        for (size_t j = 0; j < v.column(); j++) {
            os << v[i][j];
            if (j + 1 != v.column()) {
                std::cout << ", ";
            } else if (i + 1 != v.line()) {
                std::cout << std::endl;
            }
        }
    }
    return os;
}

#endif  // MATRIX_H_
