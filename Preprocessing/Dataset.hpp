#pragma once
#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include "json.hpp"

using json = nlohmann::json;

template <typename T>
using Matrix2D = std::vector<std::vector<T>>;

// class to process the dataset files, has functions to manipulate matrices
// can consider making into abstract class with virtual fucntions
template <typename T>
struct DatasetStructure
{
    Matrix2D<T> d_parsed;     // Read raw data row by row and parsed it using token file
    Matrix2D<T> d_shuffled;   // Shuffled the parsed data randomly
    Matrix2D<T> in_vector;    // Get only the input vector
    Matrix2D<T> out_vector;   // Get only the output vector
    Matrix2D<T> out_vector_s; // Output vector split
    Matrix2D<T> in_vector_t;  // Input vector transpose
    Matrix2D<T> out_vector_t; // Output vector transpose

    // Data split into 3 different sets
    // Matrix2D<T> d_training;
    // Matrix2D<T> d_validation;
    // Matrix2D<T> d_test;
};

enum DataType
{
    RAW = 0,
    SHUFFLED,
    IN,
    IN_T,
    OUT,
    OUT_S,
    OUT_T,
    TRAINING,
    VALIDATION,
    TEST
};

// @remark 
// consider making this into an abstract class for different dataset
// use template <class T> and virtual functions
class Dataset
{
public:
    Dataset(void);
    ~Dataset(void);

    void ReadDataset(const std::string &filepath, const std::string &tokenfile);
    void ExtractInOut(const unsigned int in_size);
    // void SplitDataset(const double ratio); // @todo split into train, validation, test set
    DatasetStructure<double> GetData() const { return m_data; }; // Read-Only
    void PrintData(DataType) const;                              // For Debug

private:
    DatasetStructure<double> m_data;
    void ShuffleData(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_s);
    void SplitOutput(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_o);
    void TransposeMatrix(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_t);
};
#endif