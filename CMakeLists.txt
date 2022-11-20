cmake_minimum_required(VERSION 3.0)
project(Project_0 VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set (SRC_FILES 
${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetwork/NeuralNetwork.cpp
${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetwork/Neuron.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Preprocessing/Dataset.cpp
) #source files

add_executable(Main ${SRC_FILES}) #build neural network training executable

target_include_directories(Main PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetwork
${CMAKE_CURRENT_SOURCE_DIR}/Preprocessing
) # include directory for header files
