#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>

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
    int max_iter = 80) {
    
    int n = start_point.size();
    vector<vector<double>> simplex;

    simplex.push_back(start_point);
    for (int i = 0; i < n; ++i) {
        vector<double> point = start_point;
        point[i] += simplex_size;
        simplex.push_back(point);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        sort(simplex.begin(), simplex.end(),
            [&func](const vector<double>& a, const vector<double>& b) {
                return func(a) < func(b);
            });

        double max_dist = 0.0;
        for (int i = 1; i <= n; ++i) {
            double dist = 0.0;
            for (int j = 0; j < n; ++j) {
                dist += pow(simplex[i][j] - simplex[0][j], 2);
            }
            max_dist = max(max_dist, sqrt(dist));
        }
        if (max_dist < tol) break;

        vector<double> centroid(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                centroid[j] += simplex[i][j];
            }
        }
        for (int j = 0; j < n; ++j) {
            centroid[j] /= n;
        }

        vector<double> reflected(n);
        for (int j = 0; j < n; ++j) {
            reflected[j] = centroid[j] + alpha * (centroid[j] - simplex.back()[j]);
        }
        double f_reflected = func(reflected);

        if (func(simplex[0]) <= f_reflected && f_reflected < func(simplex[n - 1])) {
            simplex.back() = reflected;
            continue;
        }

        if (f_reflected < func(simplex[0])) {
            vector<double> expanded(n);
            for (int j = 0; j < n; ++j) {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            if (func(expanded) < f_reflected) {
                simplex.back() = expanded;
            } else {
                simplex.back() = reflected;
            }
            continue;
        }

        vector<double> contracted(n);
        for (int j = 0; j < n; ++j) {
            contracted[j] = centroid[j] + rho * (simplex.back()[j] - centroid[j]);
        }
        if (func(contracted) < func(simplex.back())) {
            simplex.back() = contracted;
            continue;
        }

        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < n; ++j) {
                simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
            }
        }
    }

    return simplex[0];
}

// Метод золотого сечения
double minimize_scalar_golden(function<double(double)> func, 
                             double a, 
                             double b, 
                             double tol = 1e-6, 
                             int max_iter = 100) {
    const double phi = (1 + sqrt(5)) / 2;
    double c = b - (b - a) / phi;
    double d = a + (b - a) / phi;

    for (int i = 0; i < max_iter; ++i) {
        if (abs(c - d) < tol) break;

        if (func(c) < func(d)) {
            b = d;
        } else {
            a = c;
        }

        c = b - (b - a) / phi;
        d = a + (b - a) / phi;
    }

    return (b + a) / 2;
}

// Метод дихотомии
double minimize_scalar_dichotomy(function<double(double)> func, 
                                double a, 
                                double b, 
                                double tol = 1e-6, 
                                int max_iter = 100) {
    double epsilon = tol / 3;
    for (int i = 0; i < max_iter; ++i) {
        if (abs(b - a) < tol) break;

        double midpoint = (a + b) / 2;
        double c = midpoint - epsilon;
        double d = midpoint + epsilon;

        double fc = func(c);
        double fd = func(d);

        if (fc < fd) {
            b = d;
        } else {
            a = c;
        }
    }
    return (a + b) / 2;
}

// Модифицированный метод Розенброка
tuple<vector<double>, double, int> rosenbrock_method(
    function<double(vector<double>)> func, 
    vector<double> x0, 
    int line_search_method = 1,
    double tol = 1e-6, 
    int max_iter = 250) {

    int n = x0.size();
    vector<vector<double>> directions(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        directions[i][i] = 1.0;
    }

    vector<double> x = x0;
    int iter_num;

    for (iter_num = 0; iter_num < max_iter; ++iter_num) {
        vector<double> x_prev = x;

        for (int i = 0; i < n; ++i) {
            vector<double> direction = directions[i];

            auto line_search = [&func, &x, &direction](double alpha) {
                vector<double> point(x.size());
                for (size_t j = 0; j < x.size(); ++j) {
                    point[j] = x[j] + alpha * direction[j];
                }
                return func(point);
            };

            double alpha_min;
            if (line_search_method == 0) {
                alpha_min = minimize_scalar_dichotomy(line_search, -10.0, 10.0, tol);
            } else {
                alpha_min = minimize_scalar_golden(line_search, -10.0, 10.0, tol);
            }

            for (size_t j = 0; j < x.size(); ++j) {
                x[j] += alpha_min * direction[j];
            }
        }

        for (int i = 1; i < n; ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                directions[i][j] = x[j] - x_prev[j];
            }

            for (int j = 0; j < i; ++j) {
                double dot_product = 0.0;
                for (size_t k = 0; k < x.size(); ++k) {
                    dot_product += directions[i][k] * directions[j][k];
                }
                for (size_t k = 0; k < x.size(); ++k) {
                    directions[i][k] -= dot_product * directions[j][k];
                }
            }

            double norm = 0.0;
            for (size_t j = 0; j < x.size(); ++j) {
                norm += directions[i][j] * directions[i][j];
            }
            norm = sqrt(norm);

            if (norm > 1e-10) {
                for (size_t j = 0; j < x.size(); ++j) {
                    directions[i][j] /= norm;
                }
            }
        }

        double diff = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            diff += (x[i] - x_prev[i]) * (x[i] - x_prev[i]);
        }
        if (sqrt(diff) < tol) {
            break;
        }
    }

    double fval = func(x);
    return {x, fval, iter_num + 1};
}

// Метод случайного поиска
void random_search(
    const function<double(const vector<double>&)>& objective,
    const vector<pair<double, double>>& bounds,
    vector<double>& best_point,
    double& best_value,
    int num_samples = 1000) {
    
    random_device rd;
    mt19937 gen(rd());
    
    vector<uniform_real_distribution<>> distributions;
    for (const auto& bound : bounds) {
        distributions.emplace_back(bound.first, bound.second);
    }

    best_value = numeric_limits<double>::max();
    best_point.resize(bounds.size());

    for (int i = 0; i < num_samples; ++i) {
        vector<double> point;
        point.reserve(bounds.size());
        
        for (size_t dim = 0; dim < bounds.size(); ++dim) {
            point.push_back(distributions[dim](gen));
        }

        double value = objective(point);

        if (value < best_value) {
            best_value = value;
            best_point = point;
        }
    }
}

// Функция для вычисления суммы квадратов отклонений
double leastSquaresResidual(
    const vector<double>& a,
    const vector<double>& x,
    const vector<double>& y,
    const function<double(double, const vector<double>&)>& f) {
    
    if (x.size() != y.size() || x.empty()) {
        throw invalid_argument("Массивы данных имеют разные размерности!");
    }

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double residual = y[i] - f(x[i], a);
        sum += residual * residual;
    }
    
    return sum;
}

vector<double> calculateErrors(
    const vector<double>& params,
    const vector<double>& y,
    const vector<double>& x,
    const function<double(double, const vector<double>&)>& modelFunc,
    bool isLinearModel) {

    const int n = y.size();
    vector<double> errors;

    vector<double> y_pred(n);
    double sum_y = 0.0;
    for (int i = 0; i < n; ++i) {
        y_pred[i] = modelFunc(x[i], params);
        sum_y += y[i];
    }
    const double y_mean = sum_y / n;

    double S = 0.0;
    for (int i = 0; i < n; ++i) {
        S += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }

    const double St = sqrt(S) / n;
    double My = 0.;
    for (int i = 0; i < n; i++)
        My += y[i];
    My /= n;
    const double Sto = St / My * 100.0;

    double correlation = 0.0;
    double t_correlation = 0.0;

    if (isLinearModel) {
        double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_xy += x[i] * y[i];
            sum_x += x[i];
            sum_y += y[i];
            sum_x2 += x[i] * x[i];
            sum_y2 += y[i] * y[i];
        }
        correlation = (n * sum_xy - sum_x * sum_y) /
            sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

        if (n > 2 && correlation > -1.0 && correlation < 1.0) {
            t_correlation = correlation * sqrt((n - 2) / (1 - correlation * correlation));
        }
    }
    else {
        double suma = 0.;
        for (int i = 0; i < n; i++)
            suma += (y[i] - y_mean) * (y[i] - y_mean);
        correlation = sqrt(1 - S / suma);
    }

    errors.push_back(S);         // Суммарная квадратическая ошибка
    errors.push_back(St);        // Среднеквадратичная ошибка
    errors.push_back(Sto);       // Относительная ошибка (%)
    errors.push_back(correlation); // Коэффициент/индекс корреляции
    
    if (isLinearModel) {
        errors.push_back(t_correlation); // t-критерий 
    }

    return errors;
}

vector<double> optimizeParameters(
    // Аппроксимирующая функция
    const function<double(double, const vector<double>&)>& modelFunc,
    // Начальная точка приближения
    const vector<double>& initialGuess,
    // Вектор независимой переменной
    const vector<double>& x,
    // Вектор зависимой переменной 
    const vector<double>& y,
    // Номер метода оптимизации: 1 - Нелдера-Мида, 2 - Розенброка, 3 - Случайный поиск"
    int method,
    // Флаг линейной модели: true - линейная, false - нелинейная
    bool isLinearModel = false,
    // Точность сходимости
    double tol = 1e-6,
    // Количество итераций в методе
    int maxIter = 1000,
    // Номер метода одномерного поиска: 1 - золотое сечение, 0 - дихотомия
    int lineSearchMethod = 1,
    // Параметры для метода Нелдера-Мида:
    double simplex_size = 1.0,
    double alpha = 1.0,
    double gamma = 2.0,
    double rho = 0.5,
    double sigma = 0.5) 
{
    auto residualFunc = [&modelFunc, &x, &y](const vector<double>& a) {
        return leastSquaresResidual(a, x, y, modelFunc);
    };

    switch (method) {
        case 1: {
            
            auto result = nelder_mead(
                residualFunc, 
                initialGuess, 
                simplex_size, 
                alpha, 
                gamma, 
                rho, 
                sigma, 
                tol, 
                maxIter);

            return result;
        }
        
        case 2: {
            auto [result, S, iterations] = rosenbrock_method(
                residualFunc, initialGuess, lineSearchMethod, tol, maxIter);
            return result;
        }
        
        case 3: {
            vector<pair<double, double>> searchBounds;
            for (double val : initialGuess) {
                double range = max(abs(val), 0.1);
                searchBounds.emplace_back(val - range, val + range);
            }
            
            vector<double> best_point;
            double best_value;
            random_search(residualFunc, searchBounds, best_point, best_value, maxIter);
            return best_point;
        }
        
        default:
            throw invalid_argument("Неизвестный метод оптимизации. Допустимые значения: 1-3.");
    }
}


// Функция для решения системы линейных уравнений методом Гаусса
std::vector<double> solveGauss(std::vector<std::vector<double>> A, std::vector<double> b) {
    const size_t n = A.size();

    // Прямой ход метода Гаусса
    for (size_t k = 0; k < n; ++k) {
        // Поиск максимального элемента в столбце
        size_t max_row = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > std::abs(A[max_row][k])) {
                max_row = i;
            }
        }

        // Перестановка строк
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(b[k], b[max_row]);
        }

        // Нормализация текущей строки
        double div = A[k][k];
        if (div == 0) {
            throw std::runtime_error("Система уравнений вырождена");
        }

        for (size_t j = k; j < n; ++j) {
            A[k][j] /= div;
        }
        b[k] /= div;

        // Исключение переменной из других уравнений
        for (size_t i = 0; i < n; ++i) {
            if (i != k && A[i][k] != 0) {
                double factor = A[i][k];
                for (size_t j = k; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }

    return b;
}

// Функция вычисления коэффициентов канонического полинома
std::vector<double> fitInterpolationPolynomial(const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Размеры векторов x и y должны совпадать");
    }
    const size_t n = x.size();
    if (n == 0) {
        throw std::invalid_argument("Векторы не должны быть пустыми");
    }

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<double> b(n, 0.0);

    // Заполнение матрицы A и вектора b
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = std::pow(x[i], j);
        }
        b[i] = y[i];
    }

    // Решение системы уравнений
    return solveGauss(A, b);
}

// Функция вычисления значения полинома Лагранжа в точке x_eval
double lagrangeInterpolation(const std::vector<double>& x,
    const std::vector<double>& y,
    double x_eval) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Размеры векторов x и y должны совпадать");
    }
    if (x.empty()) {
        throw std::invalid_argument("Векторы не должны быть пустыми");
    }

    double result = 0.0;
    const size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {
        double term = y[i];
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                term *= (x_eval - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }

    return result;
}

// Функция вычисления разделенных разностей
std::vector<std::vector<double>> computeDividedDifferences(const std::vector<double>& x,
    const std::vector<double>& y) {
    const size_t n = x.size();
    std::vector<std::vector<double>> diff(n, std::vector<double>(n, 0.0));

    // Инициализация нулевого порядка (значения функции)
    for (size_t i = 0; i < n; ++i) {
        diff[i][0] = y[i];
    }

    // Вычисление разделенных разностей
    for (size_t j = 1; j < n; ++j) {
        for (size_t i = 0; i < n - j; ++i) {
            diff[i][j] = (diff[i + 1][j - 1] - diff[i][j - 1]) / (x[i + j] - x[i]);
        }
    }

    return diff;
}

// Функция построения полинома Ньютона
std::vector<double> newtonInterpolationPolynomial(const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Размеры векторов x и y должны совпадать");
    }
    if (x.empty()) {
        throw std::invalid_argument("Векторы не должны быть пустыми");
    }

    const size_t n = x.size();
    auto diff = computeDividedDifferences(x, y);
    std::vector<double> coefficients(n);

    // Коэффициенты полинома Ньютона - первые элементы каждой строки разностей
    for (size_t i = 0; i < n; ++i) {
        coefficients[i] = diff[0][i];
    }

    return coefficients;
}

// Выбор точек вокруг x_point
std::pair<std::vector<double>, std::vector<double>> select_points(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double x_point,
    size_t num_points
) {
    if (x.size() < num_points) throw std::invalid_argument("Недостаточно точек");

    auto upper = std::upper_bound(x.begin(), x.end(), x_point);
    size_t idx = upper - x.begin();

    // Корректируем индекс для центрирования вокруг x_point
    idx = std::min(std::max(num_points / 2, idx), x.size() - (num_points + 1) / 2);
    size_t start = idx - num_points / 2;

    return {
        std::vector<double>(x.begin() + start, x.begin() + start + num_points),
        std::vector<double>(y.begin() + start, y.begin() + start + num_points)
    };
}

// Основная функция
double spline_interpolation(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double x_point,
    int degree = 3
) {
    if (x.size() != y.size()) throw std::invalid_argument("Размеры x и y должны совпадать");
    if (x_point < x.front() || x_point > x.back()) throw std::invalid_argument("x_point вне интервала");

    size_t num_points = degree + 1;
    if (x.size() < num_points) throw std::invalid_argument("Слишком мало точек");

    auto return_value = select_points(x, y, x_point, num_points);
    auto& x_sub = std::get<0>(return_value);    // Первый элемент возвращаемого tuple
    auto& y_sub = std::get<1>(return_value);
    
    return lagrangeInterpolation(x_sub, y_sub, x_point);
}

double bilinearInterpolation(
    double x,
    double y,
    const std::vector<double>& x_grid,
    const std::vector<double>& y_grid,
    const std::vector<std::vector<double>>& z_values
) {
    // Проверка, что сетка не пустая
    if (x_grid.empty() || y_grid.empty()) {
        throw std::invalid_argument("Grid dimensions cannot be empty.");
    }

    // Проверка, что x_grid и y_grid отсортированы по возрастанию
    if (!std::is_sorted(x_grid.begin(), x_grid.end())) {
        throw std::invalid_argument("x_grid must be sorted in ascending order.");
    }
    if (!std::is_sorted(y_grid.begin(), y_grid.end())) {
        throw std::invalid_argument("y_grid must be sorted in ascending order.");
    }

    // Проверка, что размер z_values соответствует x_grid.size() ? y_grid.size()
    if (z_values.size() != x_grid.size()) {
        throw std::invalid_argument("z_values must have size x_grid.size() ? y_grid.size().");
    }
    for (const auto& row : z_values) {
        if (row.size() != y_grid.size()) {
            throw std::invalid_argument("z_values must have size x_grid.size() ? y_grid.size().");
        }
    }

    // Проверка, что точка (x, y) внутри сетки
    if (x < x_grid.front() || x > x_grid.back() || y < y_grid.front() || y > y_grid.back()) {
        throw std::out_of_range("Point (x, y) is outside the grid boundaries.");
    }

    // Находим индексы ближайших узлов по x (i, i+1)
    size_t i = 0;
    while (i + 1 < x_grid.size() && x_grid[i + 1] <= x) {
        i++;
    }

    // Находим индексы ближайших узлов по y (j, j+1)
    size_t j = 0;
    while (j + 1 < y_grid.size() && y_grid[j + 1] <= y) {
        j++;
    }

    // Координаты узлов
    double x0 = x_grid[i], x1 = x_grid[i + 1];
    double y0 = y_grid[j], y1 = y_grid[j + 1];

    // Значения функции в четырёх ближайших точках
    double f00 = z_values[i][j];     // f(x0, y0)
    double f01 = z_values[i][j + 1]; // f(x0, y1)
    double f10 = z_values[i + 1][j]; // f(x1, y0)
    double f11 = z_values[i + 1][j + 1]; // f(x1, y1)

    // Вычисляем веса для интерполяции
    double tx = (x - x0) / (x1 - x0);
    double ty = (y - y0) / (y1 - y0);

    // Билинейная интерполяция
    double interpolated_value =
        f00 * (1 - tx) * (1 - ty) +
        f10 * tx * (1 - ty) +
        f01 * (1 - tx) * ty +
        f11 * tx * ty;

    return interpolated_value;
}
