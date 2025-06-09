#ifndef OPTIMIZATION_LIB_H
#define OPTIMIZATION_LIB_H

#include <vector>
#include <functional>
#include <utility>
#include <tuple>
#include <gmpxx.h>


mpf_class mpf_pow(const mpf_class& base, unsigned long exp);
mpf_class mpf_pow(const mpf_class& base, int exp);

mpf_class mpf_max(const mpf_class& a, const mpf_class& b);
mpf_class mpf_max(const mpf_class& a, double b);

mpf_class mpf_exp(const mpf_class& x);

// Метод Нелдера-Мида
std::vector<mpf_class> nelder_mead(
    std::function<mpf_class(std::vector<mpf_class>)> func,
    std::vector<mpf_class> start_point,
    mpf_class simplex_size = 1.0,
    mpf_class alpha = 1.0,
    mpf_class gamma = 2.0,
    mpf_class rho = 0.5,
    mpf_class sigma = 0.5,
    mpf_class tol = 1e-6,
    int max_iter = 80);

// Метод золотого сечения
mpf_class minimize_scalar_golden(
    std::function<mpf_class(mpf_class)> func, 
    mpf_class a, 
    mpf_class b, 
    mpf_class tol = 1e-6, 
    int max_iter = 100);

// Метод дихотомии
mpf_class minimize_scalar_dichotomy(
    std::function<mpf_class(mpf_class)> func, 
    mpf_class a, 
    mpf_class b, 
    mpf_class tol = 1e-6, 
    int max_iter = 100);

// Модифицированный метод Розенброка
std::tuple<std::vector<mpf_class>, mpf_class, int> rosenbrock_method(
    std::function<mpf_class(std::vector<mpf_class>)> func, 
    std::vector<mpf_class> x0, 
    int line_search_method = 1,
    mpf_class tol = 1e-6, 
    int max_iter = 250);

// Метод случайного поиска
void random_search(
    const std::function<mpf_class(const std::vector<mpf_class>&)>& objective,
    const std::vector<std::pair<mpf_class, mpf_class>>& bounds,
    std::vector<mpf_class>& best_point,
    mpf_class& best_value,
    int num_samples = 1000);

// Функция для вычисления суммы квадратов отклонений
mpf_class leastSquaresResidual(
    const std::vector<mpf_class>& a,
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y,
    const std::function<mpf_class(mpf_class, const std::vector<mpf_class>&)>& f);

// Вычисление ошибок модели
std::vector<mpf_class> calculateErrors(
    const std::vector<mpf_class>& params,
    const std::vector<mpf_class>& y,
    const std::vector<mpf_class>& x,
    const std::function<mpf_class(mpf_class, const std::vector<mpf_class>&)>& modelFunc,
    bool isLinearModel);

// Основная функция оптимизации параметров
std::vector<mpf_class> optimizeParameters(
    const std::function<mpf_class(mpf_class, const std::vector<mpf_class>&)>& modelFunc,
    const std::vector<mpf_class>& initialGuess,
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y,
    int method,
    bool isLinearModel = false,
    mpf_class tol = 1e-6,
    int maxIter = 1000,
    int lineSearchMethod = 1,
    mpf_class simplex_size = 1.0,
    mpf_class alpha = 1.0,
    mpf_class gamma = 2.0,
    mpf_class rho = 0.5,
    mpf_class sigma = 0.5);

// Решение системы линейных уравнений методом Гаусса
std::vector<mpf_class> solveGauss(
    std::vector<std::vector<mpf_class>> A, 
    std::vector<mpf_class> b);

// Аппроксимация полиномом (каноническая форма)
std::vector<mpf_class> fitInterpolationPolynomial(
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y);

// Интерполяция методом Лагранжа
mpf_class lagrangeInterpolation(
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y,
    mpf_class x_eval);

// Вычисление разделенных разностей для полинома Ньютона
std::vector<std::vector<mpf_class>> computeDividedDifferences(
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y);

// Построение полинома Ньютона
std::vector<mpf_class> newtonInterpolationPolynomial(
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y);

// Сплайн-интерполяция
mpf_class spline_interpolation(
    const std::vector<mpf_class>& x,
    const std::vector<mpf_class>& y,
    mpf_class x_point,
    int degree = 3);

// Билинейная интерполяция
mpf_class bilinearInterpolation(
    mpf_class x,
    mpf_class y,
    const std::vector<mpf_class>& x_grid,
    const std::vector<mpf_class>& y_grid,
    const std::vector<std::vector<mpf_class>>& z_values);

#endif // OPTIMIZATION_LIB_H
