#include "Neuron.hpp"
#include "Activation.hpp"

Neuron::Neuron(unsigned int numOutputs, unsigned int index)
{
    for (auto c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_index = index;
}

void Neuron::UpdateInputWeights(Layer &prevLayer, const double &training_rate, const double &momentum)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (auto n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

        double newDeltaWeight =
            training_rate * neuron.GetOutputVal() * m_gradient + momentum * oldDeltaWeight;

        neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_index].weight += newDeltaWeight;
    }
}

void Neuron::CalcHiddenGradients(const Layer &nextLayer, int function)
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    m_gradient = sum * activateDerivative(m_outputVal, static_cast<FUNCTION>(function));
}

void Neuron::CalcOutputGradients(double targetVal, int function)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * activateDerivative(m_outputVal, static_cast<FUNCTION>(function));
}

void Neuron::FeedForward(const Layer &prevLayer, int function)
{
    double sum = 0.0;
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (auto n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].GetOutputVal() *
               prevLayer[n].m_outputWeights[m_index].weight;
    }
    m_outputVal = activate(sum, static_cast<FUNCTION>(function));
}
