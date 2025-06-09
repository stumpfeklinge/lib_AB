#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <gmpxx.h>

using namespace std;

// Вспомогательные функции
mpf_class mpf_pow(const mpf_class& base, unsigned long exp) {
    mpf_class result;
    mpf_pow_ui(result.get_mpf_t(), base.get_mpf_t(), exp);
    return result;
}

mpf_class mpf_pow(const mpf_class& base, int exp) {
    return mpf_pow(base, static_cast<unsigned long>(exp));
}

// Функция max для mpf_class
mpf_class mpf_max(const mpf_class& a, const mpf_class& b) {
    return (a > b) ? a : b;
}

// Функция max для сравнения mpf_class и double
mpf_class mpf_max(const mpf_class& a, double b) {
    return (a > mpf_class(b)) ? a : mpf_class(b);
}

// Экспанента
mpf_class mpf_exp(const mpf_class& x) {
    mpf_class sum = 1.0;
    mpf_class term = 1.0;
    mpf_class tol = 1e-20;
    unsigned long n = 1;
    
    do {
        term *= x / n;
        sum += term;
        n++;
    } while (abs(term) > tol);
    
    return sum;
}

// Метод Нелдера-Мида
vector<mpf_class> nelder_mead(
    function<mpf_class(vector<mpf_class>)> func,
    vector<mpf_class> start_point,
    mpf_class simplex_size = 1.0,
    mpf_class alpha = 1.0,
    mpf_class gamma = 2.0,
    mpf_class rho = 0.5,
    mpf_class sigma = 0.5,
    mpf_class tol = 1e-6,
    int max_iter = 80) {
    
    int n = start_point.size();
    vector<vector<mpf_class>> simplex;

    simplex.push_back(start_point);
    for (int i = 0; i < n; ++i) {
        vector<mpf_class> point = start_point;
        point[i] += simplex_size;
        simplex.push_back(point);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        sort(simplex.begin(), simplex.end(),
            [&func](const vector<mpf_class>& a, const vector<mpf_class>& b) {
                return func(a) < func(b);
            });

        mpf_class max_dist = 0.0;
        for (int i = 1; i <= n; ++i) {
            mpf_class dist = 0.0;
            for (int j = 0; j < n; ++j) {
                dist += mpf_pow(simplex[i][j] - simplex[0][j], 2);
            }
            max_dist = mpf_max(max_dist, sqrt(dist));
        }
        if (max_dist < tol) break;

        vector<mpf_class> centroid(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                centroid[j] += simplex[i][j];
            }
        }
        for (int j = 0; j < n; ++j) {
            centroid[j] /= n;
        }

        vector<mpf_class> reflected(n);
        for (int j = 0; j < n; ++j) {
            reflected[j] = centroid[j] + alpha * (centroid[j] - simplex.back()[j]);
        }
        mpf_class f_reflected = func(reflected);

        if (func(simplex[0]) <= f_reflected && f_reflected < func(simplex[n - 1])) {
            simplex.back() = reflected;
            continue;
        }

        if (f_reflected < func(simplex[0])) {
            vector<mpf_class> expanded(n);
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

        vector<mpf_class> contracted(n);
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
mpf_class minimize_scalar_golden(function<mpf_class(mpf_class)> func, 
                             mpf_class a, 
                             mpf_class b, 
                             mpf_class tol = 1e-6, 
                             int max_iter = 100) {
    const mpf_class phi = (1 + sqrt(5)) / 2;
    mpf_class c = b - (b - a) / phi;
    mpf_class d = a + (b - a) / phi;

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
mpf_class minimize_scalar_dichotomy(function<mpf_class(mpf_class)> func, 
                                mpf_class a, 
                                mpf_class b, 
                                mpf_class tol = 1e-6, 
                                int max_iter = 100) {
    mpf_class epsilon = tol / 3;
    for (int i = 0; i < max_iter; ++i) {
        if (abs(b - a) < tol) break;

        mpf_class midpoint = (a + b) / 2;
        mpf_class c = midpoint - epsilon;
        mpf_class d = midpoint + epsilon;

        mpf_class fc = func(c);
        mpf_class fd = func(d);

        if (fc < fd) {
            b = d;
        } else {
            a = c;
        }
    }
    return (a + b) / 2;
}

// Модифицированный метод Розенброка
tuple<vector<mpf_class>, mpf_class, int> rosenbrock_method(
    function<mpf_class(vector<mpf_class>)> func, 
    vector<mpf_class> x0, 
    int line_search_method = 1,
    mpf_class tol = 1e-6, 
    int max_iter = 250) {

    int n = x0.size();
    vector<vector<mpf_class>> directions(n, vector<mpf_class>(n, 0));

    for (int i = 0; i < n; ++i) {
        directions[i][i] = 1.0;
    }

    vector<mpf_class> x = x0;
    int iter_num;

    for (iter_num = 0; iter_num < max_iter; ++iter_num) {
        vector<mpf_class> x_prev = x;

        for (int i = 0; i < n; ++i) {
            vector<mpf_class> direction = directions[i];

            auto line_search = [&func, &x, &direction](mpf_class alpha) {
                vector<mpf_class> point(x.size());
                for (size_t j = 0; j < x.size(); ++j) {
                    point[j] = x[j] + alpha * direction[j];
                }
                return func(point);
            };

            mpf_class alpha_min;
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
                mpf_class dot_product = 0.0;
                for (size_t k = 0; k < x.size(); ++k) {
                    dot_product += directions[i][k] * directions[j][k];
                }
                for (size_t k = 0; k < x.size(); ++k) {
                    directions[i][k] -= dot_product * directions[j][k];
                }
            }

            mpf_class norm = 0.0;
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

        mpf_class diff = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            diff += (x[i] - x_prev[i]) * (x[i] - x_prev[i]);
        }
        if (sqrt(diff) < tol) {
            break;
        }
    }

    mpf_class fval = func(x);
    return {x, fval, iter_num + 1};
}

// Метод случайного поиска
void random_search(
    const function<mpf_class(const vector<mpf_class>&)>& objective,
    const vector<pair<mpf_class, mpf_class>>& bounds,
    vector<mpf_class>& best_point,
    mpf_class& best_value,
    int num_samples = 1000) {
    
    random_device rd;
    mt19937 gen(rd());
    
    vector<uniform_real_distribution<>> distributions;
    for (const auto& bound : bounds) {
        distributions.emplace_back(bound.first.get_d(), bound.second.get_d());
    }

    best_value = numeric_limits<mpf_class>::max();
    best_point.resize(bounds.size());

    for (int i = 0; i < num_samples; ++i) {
        vector<mpf_class> point;
        point.reserve(bounds.size());
        
        for (size_t dim = 0; dim < bounds.size(); ++dim) {
            point.push_back(mpf_class(distributions[dim](gen)));
        }

        mpf_class value = objective(point);

        if (value < best_value) {
            best_value = value;
            best_point = point;
        }
    }
}

// Функция для вычисления суммы квадратов отклонений
mpf_class leastSquaresResidual(
    const vector<mpf_class>& a,
    const vector<mpf_class>& x,
    const vector<mpf_class>& y,
    const function<mpf_class(mpf_class, const vector<mpf_class>&)>& f) {
    
    if (x.size() != y.size() || x.empty()) {
        throw invalid_argument("Массивы данных имеют разные размерности!");
    }

    mpf_class sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        mpf_class residual = y[i] - f(x[i], a);
        sum += residual * residual;
    }
    
    return sum;
}

vector<mpf_class> calculateErrors(
    const vector<mpf_class>& params,
    const vector<mpf_class>& y,
    const vector<mpf_class>& x,
    const function<mpf_class(mpf_class, const vector<mpf_class>&)>& modelFunc,
    bool isLinearModel) {

    const int n = y.size();
    vector<mpf_class> errors;

    vector<mpf_class> y_pred(n);
    mpf_class sum_y = 0.0;
    for (int i = 0; i < n; ++i) {
        y_pred[i] = modelFunc(x[i], params);
        sum_y += y[i];
    }
    const mpf_class y_mean = sum_y / n;

    mpf_class S = 0.0;
    for (int i = 0; i < n; ++i) {
        S += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }

    const mpf_class St = sqrt(S) / n;
    mpf_class My = 0.;
    for (int i = 0; i < n; i++)
        My += y[i];
    My /= n;
    const mpf_class Sto = St / My * 100.0;

    mpf_class correlation = 0.0;
    mpf_class t_correlation = 0.0;

    if (isLinearModel) {
        mpf_class sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
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
        mpf_class suma = 0.;
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

vector<mpf_class> optimizeParameters(
    const function<mpf_class(mpf_class, const vector<mpf_class>&)>& modelFunc,
    const vector<mpf_class>& initialGuess,
    const vector<mpf_class>& x,
    const vector<mpf_class>& y,
    int method,
    bool isLinearModel = false,
    mpf_class tol = 1e-6,
    int maxIter = 1000,
    int lineSearchMethod = 1,
    mpf_class simplex_size = 1.0,
    mpf_class alpha = 1.0,
    mpf_class gamma = 2.0,
    mpf_class rho = 0.5,
    mpf_class sigma = 0.5) 
{
    auto residualFunc = [&modelFunc, &x, &y](const vector<mpf_class>& a) {
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
            vector<pair<mpf_class, mpf_class>> searchBounds;
            for (mpf_class val : initialGuess) {
                mpf_class range = mpf_max(abs(val), mpf_class(0.1));
                searchBounds.emplace_back(val - range, val + range);
            }
            
            vector<mpf_class> best_point;
            mpf_class best_value;
            random_search(residualFunc, searchBounds, best_point, best_value, maxIter);
            return best_point;
        }
        
        default:
            throw invalid_argument("Неизвестный метод оптимизации. Допустимые значения: 1-3.");
    }
}

// Функция для решения системы линейных уравнений методом Гаусса
vector<mpf_class> solveGauss(vector<vector<mpf_class>> A, vector<mpf_class> b) {
    const size_t n = A.size();

    // Прямой ход метода Гаусса
    for (size_t k = 0; k < n; ++k) {
        // Поиск максимального элемента в столбце
        size_t max_row = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (abs(A[i][k]) > abs(A[max_row][k])) {
                max_row = i;
            }
        }

        // Перестановка строк
        if (max_row != k) {
            swap(A[k], A[max_row]);
            swap(b[k], b[max_row]);
        }

        // Нормализация текущей строки
        mpf_class div = A[k][k];
        if (div == 0) {
            throw runtime_error("Система уравнений вырождена");
        }

        for (size_t j = k; j < n; ++j) {
            A[k][j] /= div;
        }
        b[k] /= div;

        // Исключение переменной из других уравнений
        for (size_t i = 0; i < n; ++i) {
            if (i != k && A[i][k] != 0) {
                mpf_class factor = A[i][k];
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
vector<mpf_class> fitInterpolationPolynomial(const vector<mpf_class>& x,
    const vector<mpf_class>& y) {
    if (x.size() != y.size()) {
        throw invalid_argument("Размеры векторов x и y должны совпадать");
    }
    const size_t n = x.size();
    if (n == 0) {
        throw invalid_argument("Векторы не должны быть пустыми");
    }

    vector<vector<mpf_class>> A(n, vector<mpf_class>(n, 0.0));
    vector<mpf_class> b(n, 0.0);

    // Заполнение матрицы A и вектора b
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = mpf_pow(x[i], j);
        }
        b[i] = y[i];
    }

    // Решение системы уравнений
    return solveGauss(A, b);
}

// Функция вычисления значения полинома Лагранжа в точке x_eval
mpf_class lagrangeInterpolation(const vector<mpf_class>& x,
    const vector<mpf_class>& y,
    mpf_class x_eval) {
    if (x.size() != y.size()) {
        throw invalid_argument("Размеры векторов x и y должны совпадать");
    }
    if (x.empty()) {
        throw invalid_argument("Векторы не должны быть пустыми");
    }

    mpf_class result = 0.0;
    const size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {
        mpf_class term = y[i];
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
vector<vector<mpf_class>> computeDividedDifferences(const vector<mpf_class>& x,
    const vector<mpf_class>& y) {
    const size_t n = x.size();
    vector<vector<mpf_class>> diff(n, vector<mpf_class>(n, 0.0));

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
vector<mpf_class> newtonInterpolationPolynomial(const vector<mpf_class>& x,
    const vector<mpf_class>& y) {
    if (x.size() != y.size()) {
        throw invalid_argument("Размеры векторов x и y должны совпадать");
    }
    if (x.empty()) {
        throw invalid_argument("Векторы не должны быть пустыми");
    }

    const size_t n = x.size();
    auto diff = computeDividedDifferences(x, y);
    vector<mpf_class> coefficients(n);

    // Коэффициенты полинома Ньютона - первые элементы каждой строки разностей
    for (size_t i = 0; i < n; ++i) {
        coefficients[i] = diff[0][i];
    }

    return coefficients;
}

// Выбор точек вокруг x_point
pair<vector<mpf_class>, vector<mpf_class>> select_points(
    const vector<mpf_class>& x,
    const vector<mpf_class>& y,
    mpf_class x_point,
    size_t num_points
) {
    if (x.size() < num_points) throw invalid_argument("Недостаточно точек");

    auto upper = upper_bound(x.begin(), x.end(), x_point);
    size_t idx = upper - x.begin();

    // Корректируем индекс для центрирования вокруг x_point
    idx = min(max(num_points / 2, idx), x.size() - (num_points + 1) / 2);
    size_t start = idx - num_points / 2;

    return {
        vector<mpf_class>(x.begin() + start, x.begin() + start + num_points),
        vector<mpf_class>(y.begin() + start, y.begin() + start + num_points)
    };
}

// Основная функция
mpf_class spline_interpolation(
    const vector<mpf_class>& x,
    const vector<mpf_class>& y,
    mpf_class x_point,
    int degree = 3
) {
    if (x.size() != y.size()) throw invalid_argument("Размеры x и y должны совпадать");
    if (x_point < x.front() || x_point > x.back()) throw invalid_argument("x_point вне интервала");

    size_t num_points = degree + 1;
    if (x.size() < num_points) throw invalid_argument("Слишком мало точек");

    auto return_value = select_points(x, y, x_point, num_points);
    auto& x_sub = get<0>(return_value);
    auto& y_sub = get<1>(return_value);
    
    return lagrangeInterpolation(x_sub, y_sub, x_point);
}

mpf_class bilinearInterpolation(
    mpf_class x,
    mpf_class y,
    const vector<mpf_class>& x_grid,
    const vector<mpf_class>& y_grid,
    const vector<vector<mpf_class>>& z_values
) {
    // Проверка, что сетка не пустая
    if (x_grid.empty() || y_grid.empty()) {
        throw invalid_argument("Grid dimensions cannot be empty.");
    }

    // Проверка, что x_grid и y_grid отсортированы по возрастанию
    if (!is_sorted(x_grid.begin(), x_grid.end())) {
        throw invalid_argument("x_grid must be sorted in ascending order.");
    }
    if (!is_sorted(y_grid.begin(), y_grid.end())) {
        throw invalid_argument("y_grid must be sorted in ascending order.");
    }

    // Проверка, что размер z_values соответствует x_grid.size() ? y_grid.size()
    if (z_values.size() != x_grid.size()) {
        throw invalid_argument("z_values must have size x_grid.size() ? y_grid.size().");
    }
    for (const auto& row : z_values) {
        if (row.size() != y_grid.size()) {
            throw invalid_argument("z_values must have size x_grid.size() ? y_grid.size().");
        }
    }

    // Проверка, что точка (x, y) внутри сетки
    if (x < x_grid.front() || x > x_grid.back() || y < y_grid.front() || y > y_grid.back()) {
        throw out_of_range("Point (x, y) is outside the grid boundaries.");
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
    mpf_class x0 = x_grid[i], x1 = x_grid[i + 1];
    mpf_class y0 = y_grid[j], y1 = y_grid[j + 1];

    // Значения функции в четырёх ближайших точках
    mpf_class f00 = z_values[i][j];     // f(x0, y0)
    mpf_class f01 = z_values[i][j + 1]; // f(x0, y1)
    mpf_class f10 = z_values[i + 1][j]; // f(x1, y0)
    mpf_class f11 = z_values[i + 1][j + 1]; // f(x1, y1)

    // Вычисляем веса для интерполяции
    mpf_class tx = (x - x0) / (x1 - x0);
    mpf_class ty = (y - y0) / (y1 - y0);

    // Билинейная интерполяция
    mpf_class interpolated_value =
        f00 * (1 - tx) * (1 - ty) +
        f10 * tx * (1 - ty) +
        f01 * (1 - tx) * ty +
        f11 * tx * ty;

    return interpolated_value;
}
