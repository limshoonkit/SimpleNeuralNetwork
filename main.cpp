#include <iostream>
#include <string>

#include "NeuralNetwork.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Command not recognize!" << std::endl
                  << "Syntax:" << std::endl;
        std::cout << ".\\Main.exe [Config]" << std::endl;
        exit(-1);
    }
    const std::string configFile = argv[1];
    // create a unique pointer for the NeuralNetwork and Dataset class
    auto nn = std::make_unique<NeuralNetwork>(configFile);
    // print to console information of the neural network
    // nn->PrintConfig();
    // nn->PrintDataset(SHUFFLED);
    // nn->PrintDataset(IN);
    // nn->PrintDataset(OUT);
    // nn->PrintDataset(OUT_S);
    nn->Train();
}