#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

class Neuron;

struct Connection
{
    double weight = 0;
    double deltaWeight = 0;
};

typedef std::vector<Neuron> Layer;

/* @brief
 *   Main class for each individual neurons
 *   Takes in the number of output and its index in the layer
 */
class Neuron
{
public:
    Neuron(unsigned int numOutputs, unsigned int index);
    inline void SetOutputVal(double val) { m_outputVal = val; }
    inline double GetOutputVal(void) const { return m_outputVal; }
    void FeedForward(const Layer &prevLayer, int function);
    void CalcOutputGradients(double targetVal, int function);
    void CalcHiddenGradients(const Layer &nextLayer, int function);
    void UpdateInputWeights(Layer &prevLayer, const double &training_rate, const double &momentum);

private:

    inline double randomWeight(void) { return rand() / double(RAND_MAX); }
    double m_outputVal = 0;
    std::vector<Connection> m_outputWeights{};
    unsigned int m_index = 0U;
    double m_gradient = 0;
};

#endif