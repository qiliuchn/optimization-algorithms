// unconstrained optimization, steepest descent method
// Qi Liu, liuqi_tj@hotmail.com
//
// Wolfe condtion is used for line search methods
// Cf. Nocedal, Wright, p33
// The way to find step size that satisfies Wolfe condition: Nocedal, p60
#include <iostream>
#include <cmath>
using namespace std;

// function prototypes
void fminunc_steepest_descent(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);
void copy(int dim, double source[], double dest[]);
void copy_and_negate(int dim, double source[], double dest[]);
double l2norm(int dim, double v[]);
double inner_produce(int dim, double x[], double y[]);
void x_plus_c_multi_y(int dim, double x[], double c, double y[], double output[]);
double phi(int dim,
            double (*f)(double x[]),
            double x[],
            double direct[],
            double alpha);
double phi_derivative(int dim,
                        void (*gradient)(double x[], double grad[2]),
                        double x[],
                        double direct[],
                        double alpha);
double zoom(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1,
            double c2,
            double alpha_lo,
            double alpha_hi);                   

void fminunc_steepest_descent(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    // @arg obj: objective function
    // @arg gradent: gradient function
    // @arg dim: dimension of input
    // @arg x0: starting point
    // @arg x: return x
    // @arg fmin: return minimum f
    // @return true if successfully find the value
    copy(dim, x0, x); // initlize x
    double grad[2];  // store gradient at x
    double direct[2]; // store moving direction at x
    double threshold_x = 0.0001; // threshold for stopping x iter
    int max_iter_x = 100000; // max num of x iterations

    // start iteration on x
    int i_x = 1;
    while (true) {   
        // evaluate f
        fmin = (*f)(x);

        // find gradient, grad
        (*gradient)(x, grad);
        // update search direction - negative gradient, direct
        copy_and_negate(dim, grad, direct);
        
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;

        if (i_x == max_iter_x) {
            cerr << "Not converging! Error code: 01" << endl;
            exit(1);
        } 

        // find step size - alpha*
        // alpha related params
        double c1 = 0.0001; // c1 setting, cf p 33, "loose line search" settings, p62
        double c2 = 0.9;  // "loose line search" settings, p62
        double alpha_max = 4; // max step size
        double alpha_0 = 0;
        double phi_alpha_0 = phi(dim, f, x, direct, 0);
        double phi_deriv_alpha_0 = phi_derivative(dim, gradient, x, direct, 0);
        double alpha_prev = alpha_0; // used for iteration
        double phi_alpha_prev = phi_alpha_0;
        double alpha = 0.5 * alpha_max;  // used for iteration
        double phi_alpha;
        double phi_deriv_alpha;
        double alpha_star; // used to store final step size; introduced purely to match notations in Nocedal.
        int max_iter_alpha = 10000; // max number of alpha iterations

        // start iteration on alpha
        int i_alpha = 1;
        while (true) {
            // check terminating condition
            if (i_alpha == max_iter_alpha) {
                cerr << "Line search fails! Error code: 02" << endl;
                exit(2);
            }

            // evaluate phi(alpha)
            phi_alpha = phi(dim, f, x, direct, alpha);
            if ((phi_alpha > (phi_alpha_0 + c1 * phi_deriv_alpha_0))
                || (phi_alpha >= phi_alpha_prev && i_alpha > 1)) {
                    alpha_star = zoom(dim, f, gradient, x, direct, c1, c2, alpha_prev, alpha);
                    break;
            }
            // evaluate phi deriv alpha
            phi_deriv_alpha = phi_derivative(dim, gradient, x, direct, alpha);
            if (abs(phi_deriv_alpha) <= (-1) * c2 * phi_deriv_alpha_0) {
                alpha_star = alpha;
                break;
            }

            if (phi_deriv_alpha >= 0) {
                alpha_star = zoom(dim, f, gradient, x, direct, c1, c2, alpha, alpha_prev);
                break;
            }

            // update alpha, alpha_prev
            alpha_prev = alpha;
            alpha = (alpha + alpha_max) / 2;     
            ++i_alpha;
        }
        
        // update x
        for (int i = 0; i < dim; ++i) {
           x[i] = x[i] + alpha_star * direct[i];
        }
        ++i_x;
    }
}

double zoom(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1,
            double c2,
            double alpha_lo,
            double alpha_hi) {
    // Nocedal, p61
    double alpha;
    double phi_alpha;
    double phi_alpha_0 = phi(dim, f, x, direct, 0);
    double phi_deriv_alpha_0 = phi_derivative(dim, gradient, x, direct, 0);
    double phi_alpha_lo;
    double phi_deriv_alpha;
    int max_itr = 1000;

    // start iteration
    int i = 0;
    while (true) {
        if (i == max_itr) {
            cerr << "Zoom fails! Error code: 03" << endl;
            exit(3);
        }

        phi_alpha_lo = phi(dim, f, x, direct, alpha_lo);
        // interpolate
        alpha = (alpha_lo + alpha_hi) / 2;  // bisection
        // evaluate phi_alpha
        phi_alpha = phi(dim, f, x, direct, alpha);
        if (phi_alpha > (phi_alpha_0 + c1 * alpha * phi_deriv_alpha_0) || phi_alpha >= phi_alpha_lo) {
            alpha_hi = alpha;
        } else {
            phi_deriv_alpha = phi_derivative(dim, gradient, x, direct, alpha);
            if (abs(phi_deriv_alpha) <= (-1) * c2 * phi_deriv_alpha_0) {
                return alpha;
            }

            if (phi_deriv_alpha * (alpha_hi - alpha_lo) >= 0) {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha;
        }
        ++i;
    }
}

void copy(int dim, double source[], double dest[]) {
    for (int i = 0; i < dim; ++i)
        dest[i] = source[i];
}

void copy_and_negate(int dim, double source[], double dest[]) {
    for (int i = 0; i < dim; ++i)
        dest[i] = (-1) * source[i];
}

double l2norm(int dim, double v[]) {
    /**
     * @brief L2-norm of vector v with dimension dim
     */
    double ans = 0;

    for (int i = 0; i < dim; ++i) {
        ans += pow(v[i], 2);
    }
    return sqrt(ans);
}

double inner_produce(int dim, double x[], double y[]) {
    double ans = 0;

    for (int i = 0; i < dim; ++i) {
        ans += x[i] * y[i];
    }
    return ans;
}

void x_plus_c_multi_y(int dim, double x[], double c, double y[], double output[]) {
    /**
     * @brief compute x + cy and store in ans;
     * x and y are vectors of dimensiom dim; c is a scalar; 
     */
    for (int i = 0; i < dim; ++i)
        output[i] = x[i] + c * y[i];   
}

double phi(int dim,
            double (*f)(double x[]),
            double x[],
            double direct[],
            double alpha)
{
    double x_alpha[dim];

    x_plus_c_multi_y(dim, x, alpha, direct, x_alpha);
    return (*f)(x_alpha);
}

double phi_derivative(int dim,
                        void (*gradient)(double x[], double grad[2]),
                        double x[],
                        double direct[],
                        double alpha) {
    double x_alpha[dim];
    double grad_x_alpha[dim];

    x_plus_c_multi_y(dim, x, alpha, direct, x_alpha);
    (*gradient)(x_alpha, grad_x_alpha);
    return inner_produce(dim, direct, grad_x_alpha);
}
