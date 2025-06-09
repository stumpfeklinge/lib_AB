#ifndef AB
#define AB

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>

using namespace std;

// Метод Нелдера-Мида
vector<double> nelder_mead(
    function<double(vector<double>)> func,
    vector<double> start_point,
    double simplex_size = 1.0,
    double alpha = 1.0,
    double gamma = 2.0,
    double rho = 0.5,
    double sigma = 0.5,
    double tol = 1e-6,
    int max_iter = 80);

// Метод золотого сечения
double minimize_scalar_golden(function<double(double)> func, 
                             double a, 
                             double b, 
                             double tol = 1e-6, 
                             int max_iter = 100);

// Метод дихотомии
double minimize_scalar_dichotomy(function<double(double)> func, 
                                double a, 
                                double b, 
                                double tol = 1e-6, 
                                int max_iter = 100);

// Модифицированный метод Розенброка
tuple<vector<double>, double, int> rosenbrock_method(
    function<double(vector<double>)> func, 
    vector<double> x0, 
    int line_search_method = 1,
    double tol = 1e-6, 
    int max_iter = 250);

// Метод случайного поиска
void random_search(
    const function<double(const vector<double>&)>& objective,
    const vector<pair<double, double>>& bounds,
    vector<double>& best_point,
    double& best_value,
    int num_samples = 1000);

// Функция для вычисления суммы квадратов отклонений
double leastSquaresResidual(
    const vector<double>& a,
    const vector<double>& x,
    const vector<double>& y,
    const function<double(double, const vector<double>&)>& f);

// Функция для расчета ошибок модели
vector<double> calculateErrors(
    const vector<double>& params,
    const vector<double>& y,
    const vector<double>& x,
    const function<double(double, const vector<double>&)>& modelFunc,
    bool isLinearModel);

// Основная функция оптимизации параметров
vector<double> optimizeParameters(
    const function<double(double, const vector<double>&)>& modelFunc,
    const vector<double>& initialGuess,
    const vector<double>& x,
    const vector<double>& y,
    int method,
    bool isLinearModel = false,
    double tol = 1e-6,
    int maxIter = 1000,
    int lineSearchMethod = 1,
    double simplex_size = 1.0,
    double alpha = 1.0,
    double gamma = 2.0,
    double rho = 0.5,
    double sigma = 0.5);

#endif // OPTIMIZATION_METHODS_H
/**
 * @brief Находит коэффициенты интерполяционного полинома
 * 
 * @param x Вектор значений x
 * @param y Вектор значений y
 * @return std::vector<double> Коэффициенты полинома
 */
std::vector<double> fitInterpolationPolynomial(
    const std::vector<double>& x, 
    const std::vector<double>& y);

/**
 * @brief Интерполяция Лагранжа
 * 
 * @param x Узлы интерполяции
 * @param y Значения в узлах
 * @param x_eval Точка вычисления
 * @return double Значение полинома
 */
double lagrangeInterpolation(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double x_eval);

/**
 * @brief Полином Ньютона
 * 
 * @param x Узлы интерполяции
 * @param y Значения в узлах
 * @return std::vector<double> Коэффициенты полинома
 */
std::vector<double> newtonInterpolationPolynomial(
    const std::vector<double>& x,
    const std::vector<double>& y);

/**
 * @brief Сплайн-интерполяция
 * 
 * @param x Узлы интерполяции
 * @param y Значения в узлах
 * @param x_point Точка вычисления
 * @param degree Степень сплайна (1-3)
 * @return double Значение сплайна
 */
double spline_interpolation(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double x_point,
    int degree = 3);

/**
 * @brief Билинейная интерполяция
 * 
 * @param x x-координата точки
 * @param y y-координата точки
 * @param x_grid Сетка по x
 * @param y_grid Сетка по y
 * @param z_values Значения функции
 * @return double Интерполированное значение
 */
double bilinearInterpolation(
    double x,
    double y,
    const std::vector<double>& x_grid,
    const std::vector<double>& y_grid,
    const std::vector<std::vector<double>>& z_values);
