#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include "Dataset.hpp"

using TokenMap = std::map<std::string, std::string>;

Dataset::Dataset()
{
}

Dataset::~Dataset()
{
}

void Dataset::ReadDataset(const std::string &filepath, const std::string &tokenfile)
{
    std::cout << "Filepath : " << filepath << std::endl;
    std::ifstream fs(filepath);
    std::string line = "";
    std::ifstream ts(tokenfile);
    TokenMap m{};
    std::vector<double> temp;

    if (fs.is_open())
    {
        // dataset has to replace non-numeric expressions
        if (ts.is_open())
        {
            json j = json::parse(ts);
            for (auto &token : j["token"])
                m.insert(std::make_pair(token["name"].get<std::string>(), token["value"].get<std::string>()));
        }
        ts.close(); // remember to close file to prevent leak
        while (std::getline(fs, line))
        {
            // replace the tokens defined in json
            if (!m.empty())
                for (const auto &[k, v] : m)
                {
                    size_t pos = line.find(k); // return -1 if fail
                    if (pos != -1)             // if found occurance
                        line.replace(pos, k.length(), v);
                    pos = -1;
                }
            // replace ',' with whitespace so that sstream can parse word by word
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            std::copy(std::istream_iterator<double>(iss),
                      std::istream_iterator<double>(),
                      std::back_inserter(temp)); // copy the data line by line into a temp container
            m_data.d_parsed.emplace_back(temp);  // store the data in the class container
            temp.clear();                        // remmeber to clear the temp container
        }
        fs.close(); // remember to close file to prevent leak
        ShuffleData(m_data.d_parsed, m_data.d_shuffled);
        return;
    }

    std::cout << "Unable to open file" << std::endl;
    exit(EXIT_FAILURE);
}

void Dataset::PrintData(DataType type) const
{
    Matrix2D<double> m_temp{};
    std::string s_temp = "";
    switch (type)
    {
    case RAW:
        s_temp = "RAW";
        m_temp = m_data.d_parsed;
        break;
    case SHUFFLED:
        s_temp = "SHUFFLED";
        m_temp = m_data.d_shuffled;
        break;
    case IN:
        s_temp = "INPUT";
        m_temp = m_data.in_vector;
        break;
    case OUT:
        s_temp = "OUTPUT";
        m_temp = m_data.out_vector;
        break;
    case OUT_S:
        s_temp = "OUTPUT SPLITTED";
        m_temp = m_data.out_vector_s;
        break;
    case IN_T:
        s_temp = "INPUT_TRANSPOSE";
        m_temp = m_data.in_vector_t;
        break;
    case OUT_T:
        s_temp = "OUTPUT_TRANSPOSE";
        m_temp = m_data.out_vector_t;
        break;
    default:
        std::cout << "No Data Type Specified." << std::endl;
        return;
    }

    unsigned int index = 1U;
    std::cout << "[Index] \t| Parsed Data (" << s_temp << ")" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    for (auto &row : m_temp)
    {
        std::cout << "[" << index++ << "]\t| ";
        for (auto &entry : row)
            std::cout << std::setprecision(2) << entry << " | ";
        std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------" << std::endl;
}

// get the transpose for input and output vector
// assuming that the output is positioned after the inputs then knowing the input size is enough
void Dataset::ExtractInOut(const unsigned int in_size)
{
    // sanity check .. use shuffled dataset
    if (m_data.d_shuffled.empty())
        return;

    m_data.in_vector = m_data.d_shuffled;
    for (auto &row : m_data.in_vector)
        row.erase(row.begin() + in_size, row.end());
    TransposeMatrix(m_data.in_vector, m_data.in_vector_t);

    m_data.out_vector = m_data.d_shuffled;
    for (auto &row : m_data.out_vector)
        row.erase(row.begin(), row.begin() + in_size);
    SplitOutput(m_data.out_vector, m_data.out_vector_s);
    TransposeMatrix(m_data.out_vector, m_data.out_vector_t);
}

// private functions
void Dataset::ShuffleData(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_s)
{
    // sanity check
    if (matrix.empty())
        return;

    matrix_s = matrix;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(matrix_s.begin(), matrix_s.end(), g); // random shuffle the dataset
}

void Dataset::TransposeMatrix(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_t)
{
    std::vector<double> temp;
    for (auto i = 0; i < matrix.front().size(); i++)
    {
        for (auto &row : matrix)
            temp.emplace_back(row[i]);
        matrix_t.emplace_back(temp);
        temp.clear();
    }
}

void Dataset::SplitOutput(const Matrix2D<double> &matrix, Matrix2D<double> &matrix_o)
{
    unsigned int maxVal = 0U;
    for (auto &val : matrix)
    {
        maxVal = maxVal > val[0] ? maxVal : val[0];
    }
    if (maxVal == 0)
        return;

    std::vector<double> temp;
    for (auto &val : matrix)
    {
        for (auto i = 0; i < maxVal + 1; ++i)
            temp.emplace_back((val[0] == i) ? 1 : 0);
        matrix_o.emplace_back(temp);
        temp.clear();
    }
}
