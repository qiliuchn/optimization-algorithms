// test case of minimizing f - unconstrained
// Qi Liu, liuqi_tj@hotmail.com
//
// Three methods are tested:
// - Steepest descent
// - Newton direaction with Hessian modification
// - Quasinewton method - BFGS
#include <iostream>
#include <cmath>
using namespace std;

void fminunc_steepest_descent(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);

double f(double x[]) {
    // test function
    // @arg dim: input dimension
    // @arg x: input array
    return x[0] * x[0] + sin(x[1]);
}

void gradient(double x[], double grad[]) {
    // gradient function is provided algorithms that require it
    grad[0] = 2 * x[0];
    grad[1] = cos(x[1]);
}

void hessian(double x[], double hess[][2]) {
    // gradient function is provided algorithms that require it
    hess[0][0] = 2;
    hess[0][1] = 0;
    hess[1][0] = 0;
    hess[1][1] = - x[1];
}

int main(int argc, char **argv) {
    // for test only
    int dim = 2;
    double x0[2] = {0.5, 0.5};
    double x[2];
    double fmin;

    fminunc_steepest_descent(dim, &f, &gradient, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl;
    return 0;
}

