#include <fstream>
#include <iostream>
#include "NeuralNetwork.hpp"

void NeuralNetwork::ParseConfig()
{
    json config;
    // read in the json file
    std::ifstream f(m_configPath, std::ifstream::in);
    // initialize json object with what was read from file
    if (!f.is_open())
    {
        std::cerr << "Failed to open file! Config Path : " << m_configPath << std::endl;
        exit(-1);
    }
    std::cout << "Config Path : " << m_configPath << std::endl;
    f >> config;
    if (config.size() < NUM_CONFIG)
    {
        std::cerr << "Config Mismatched! Expected" << NUM_CONFIG << "Elements, Found " << config.size() << "." << std::endl;
        exit(-1);
    }
    f.close(); // remember to close file to prevent leak
    // assign the network config values
    m_config = NetworkConfig{
        .datasetPath = config["datasetPath"],
        .tokenPath = config["tokenPath"],
        .importWeightPath = config["importWeightPath"],
        .exportWeightPath = config["exportWeightPath"],
        .topology = config["topology"],
        .training_split = config["training_split"],
        .learning_rate = config["learning_rate"],
        .momentum = config["momentum"],
        .bias = config["bias"],
        .activationFunction = config["hiddenLayerActivation"],
        .epoch = config["epoch"],
        .accuracyThreshold = config["accuracyThreshold"]
        // @todo add additional hyperparameters
        //  .isBatchLearning = config["isBatchLearning"],
        //  .batchsize = config["batchSize"],
        //  .isRegularized = config["isRegularized"],
        //  .regularizationRate = config["regularizationRate"]
    };
    // read in the dataset file
    ptr_ds->ReadDataset(m_config.datasetPath, m_config.tokenPath);
    // extract the input and output datasets
    ptr_ds->ExtractInOut(m_config.topology[0]);
}

void NeuralNetwork::Train()
{
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Training started! " << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    InitNetwork();
    unsigned int training_pass = 1U;
    // @todo normalized input value
    Matrix2D<double> in = ptr_ds->GetData().in_vector;
    // @remark
    // use the splitted output vector
    // should also use softmax function for output layer activation
    Matrix2D<double> out = ptr_ds->GetData().out_vector_s;
    if (in.size() != out.size())
    {
        std::cerr << "Input size not match output size! " << std::endl;
        exit(-1);
    }
    unsigned int size = in.size();
    m_recentAverageError = 0;
    // either epoch ended or accuracy threshold reached after certain % of epoch
    while (training_pass < m_config.epoch)
    {
        std::cout << "Training Pass: " << training_pass << std::endl;
        for (auto i = 0; i < size; ++i)
        {
            FeedForward(in[i]);
            PrintIntermediateOutput(out[i]);
            BackPropagate(out[i]);
            // Report how well the training is working, average over recent samples
            std::cout << "Avg error: " << m_recentAverageError << std::endl;
        }
        training_pass++;
        if ((double)training_pass / (double)m_config.epoch > 0.25 && (1 - m_recentAverageError > m_config.accuracyThreshold))
            break; // threshold termination
    }
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Training ended at Epoch " << training_pass << " with Error of " << m_recentAverageError << "." << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
}

// @todo check for bias flag
void NeuralNetwork::InitNetwork()
{
    // sanity check
    if (m_config.topology.empty())
    {
        std::cerr << "Topology not recognized! " << std::endl;
        exit(-1);
    }

    unsigned int layerSize = m_config.topology.size();
    for (auto index_layer = 0; index_layer < layerSize; ++index_layer)
    {
        m_network.emplace_back(Layer());
        auto numOutput = (index_layer == layerSize - 1) ? 0 : m_config.topology[index_layer + 1];
        // Add a bias neuron in each layer.
        for (auto index_neuron = 0; index_neuron <= m_config.topology[index_layer]; ++index_neuron)
            m_network.back().emplace_back(Neuron(numOutput, index_neuron));
    }
    // Force the bias node's output to a value
    m_network.back().back().SetOutputVal(m_config.bias);
}

void NeuralNetwork::FeedForward(const std::vector<double> &in)
{
    // sanity check
    if (in.size() != m_network.front().size() - 1)
    {
        std::cerr << "Input size mismatched! " << std::endl;
        exit(-1);
    }
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < in.size(); ++i)
        m_network[0][i].SetOutputVal(in[i]);

    // forward propagate
    for (auto index_layer = 1; index_layer < m_network.size(); ++index_layer)
    {
        Layer &prevLayer = m_network[index_layer - 1];
        for (auto n = 0; n < m_network[index_layer].size() - 1; ++n)
            m_network[index_layer][n].FeedForward(prevLayer, m_config.activationFunction);
    }
}

void NeuralNetwork::BackPropagate(const std::vector<double> &out)
{
    // sanity check
    if (out.size() != m_network.back().size() - 1)
    {
        std::cerr << "Output size mismatched! " << std::endl;
        exit(-1);
    }
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_network.back();
    m_error = 0.0;

    for (auto n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = out[n] - outputLayer[n].GetOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);           // RMS

    // Implement a recent average measurement
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (auto m = 0; m < outputLayer.size() - 1; ++m)
        outputLayer[m].CalcOutputGradients(out[m], m_config.activationFunction);

    // Calculate hidden layer gradients
    for (auto index_layer = m_network.size() - 2; index_layer > 0; --index_layer)
    {
        Layer &hiddenLayer = m_network[index_layer];
        Layer &nextLayer = m_network[index_layer + 1];
        for (auto i = 0; i < hiddenLayer.size(); ++i)
            hiddenLayer[i].CalcHiddenGradients(nextLayer, m_config.activationFunction);
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (auto index_layer = m_network.size() - 1; index_layer > 0; --index_layer)
    {
        Layer &layer = m_network[index_layer];
        Layer &prevLayer = m_network[index_layer - 1];
        for (auto j = 0; j < layer.size() - 1; ++j)
            layer[j].UpdateInputWeights(prevLayer, m_config.learning_rate, m_config.momentum);
    }
}

void NeuralNetwork::PrintIntermediateOutput(const std::vector<double> &out) const
{
    for (auto i = 0; i < m_network.back().size() - 1; ++i)
    {
        std::cout << std::setprecision(4) << "Predict(" << m_network.back()[i].GetOutputVal() << ") Actual(" << out[i] << ")\t";
    }
}

void NeuralNetwork::PrintConfig() const
{
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Neural Network Configuration" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Dataset \t: " << m_config.datasetPath << std::endl;
    std::cout << "Token File \t: " << m_config.tokenPath << std::endl;
    std::cout << "Import Weight \t: " << m_config.importWeightPath << std::endl;
    std::cout << "Export Weight \t: " << m_config.exportWeightPath << std::endl;
    std::cout << "Topology \t: [ ";
    for (auto layer : m_config.topology)
        std::cout << layer << " ";
    std::cout << "]" << std::endl;
    std::cout << "Training Split \t: " << m_config.training_split << std::endl;
    std::cout << "Learning Rate \t: " << m_config.learning_rate << std::endl;
    std::cout << "Momentum \t: " << m_config.momentum << std::endl;
    std::cout << "Bias Value\t: " << m_config.bias << std::endl;
    std::cout << "Activation \t: " << m_config.activationFunction << " (0:Sigmoid , 1:Tanh, 2:ReLu, 3:Linear)" << std::endl;
    std::cout << "Epoch \t\t: " << m_config.epoch << std::endl;
    std::cout << "Threshold \t: " << m_config.accuracyThreshold << std::endl;
    // std::string isBatch = (m_config.isBatchLearning) ? "Yes" : "No";
    // std::cout << "Batch Learning \t: " << isBatch << std::endl;
    // std::cout << "Batch Size \t: " << m_config.batchsize << std::endl;
    // std::string isRegularized = (m_config.isRegularized) ? "Yes" : "No";
    // std::cout << "Regularized \t: " << isRegularized << std::endl;
    // std::cout << "Reg Rate \t: " << m_config.regularizationRate << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
}