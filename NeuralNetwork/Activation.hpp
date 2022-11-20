#pragma once
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <algorithm>

enum FUNCTION
{
    SIGMOID = 0,
    TANH,
    RELU,
    LINEAR
};

double activate(double x, FUNCTION func)
{
    switch (func)
    {
    case RELU:
        return std::max(x, 0.0);
    case TANH:
        return std::tanh(x);
    case SIGMOID:
        return (1 / (1 + std::exp(-x)));
    default:
        return x; // Essentially reduce to linear regression
    }
}

double activateDerivative(double x, FUNCTION func)
{
    switch (func)
    {
    case RELU:
        return x > 0;
    case TANH:
        return 1 - x * x;
    case SIGMOID:
        return x * (1 - x);
    default:
        return 1;
    }
}
#endif