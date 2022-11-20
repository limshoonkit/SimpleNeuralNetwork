#pragma once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <vector>
#include "json.hpp"
#include "Neuron.hpp"
#include "Dataset.hpp"

using json = nlohmann::json;

#define NUM_CONFIG 12

struct NetworkConfig
{
    // Default config values
    std::string datasetPath = "/Dataset/dataset.csv";
    std::string tokenPath = "/Dataset/dataset.json";
    std::string importWeightPath = "/Trained/weights.csv";
    std::string exportWeightPath = "/Trained/weights.csv";
    std::vector<unsigned short> topology{2, 2, 2};
    double training_split = 0.15;
    double learning_rate = 0.15;
    double momentum = 0.5;
    double bias = 0.5;
    unsigned short activationFunction = 0U;
    unsigned short epoch = 1000U; // short so epoch capped at 65535
    double accuracyThreshold = 0.85;
    // bool isBatchLearning = false;
    // unsigned short batchsize = 10U;
    // bool isRegularized = false;
    // double regularizationRate = 0.5;
};

/* @brief
 *   Main class for the neural network
 *   Takes in configuration parameters and a pointer to the dataset
 */
class NeuralNetwork
{
public:
    NeuralNetwork(const std::string &path) : m_configPath(path)
    {
        ptr_ds = std::make_unique<Dataset>();
        ParseConfig();
    };

    // Core Functionsk
    void Train(); // other context may call it Fit()
    // @todo predict and export weights
    void Predict();
    void ExportWeights();

    // Utility Functions
    void PrintConfig() const;                                                   // Debugging, Read-Only
    inline void PrintDataset(DataType type) const { ptr_ds->PrintData(type); }; // Debugging, Read-Only
    void PrintIntermediateOutput(const std::vector<double> &out) const;         // Debugging, Read-Only

    // Getters & Setter
    inline NetworkConfig GetConfig() const { return m_config; };

private:
    std::string m_configPath = "";
    NetworkConfig m_config{};
    std::unique_ptr<Dataset> ptr_ds = nullptr;

    Matrix2D<Neuron> m_network; // m_network[layerIndex][neuronIndex]
    double m_error = 0.0;
    double m_recentAverageError = 0.0;
    const double m_recentAverageSmoothingFactor = 100;

    void ParseConfig();
    void InitNetwork();
    void FeedForward(const std::vector<double> &in);
    void BackPropagate(const std::vector<double> &out);
};

#endif