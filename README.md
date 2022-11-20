**[NOTE: WIP. Not fully tested, there are still bugs presesnt.]**

# C++ implementation of a Back Propagation Neural Network

Implementation of a simple neural network using C++
1. Dataset - Folder to store the datasets for training
2. Neural Network - Core classes for the neural network
3. Preprocessing - Utility classes to process the dataset

Credits to https://www.millermattson.com/dave/?p=54

## Dataset

There are two sample datasets. The classic iris dataset https://www.kaggle.com/datasets/uciml/iris and a simplified breast cancer dataset https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset.

Iris dataset has 4 inputs and 3 output class.
Input
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
Output
* Iris-setosa       (1 0 0)
* Iris-versicolor   (0 1 0)
* Iris-virginica    (0 0 1)

Breast cancer dataset has 8 inputs and 2 output class.
Input
* Radius of Lobes
* Mean of Surface Texture
* Outer Perimeter of Lobes
* Mean Area of Lobes
* Mean of Smoothness Levels
* Mean of Compactness
* Mean of Concavity
* Mean of Cocave Points
Output
* Benign       (1 0)
* Malignant    (0 1)

## Build/Compiling

* You need either cmake or Visual Studio Code, build configurations are in CMakeLists.txt

From cmd line,
```cs
mkdir build
cmake --build "/build"
```

For VS Code, install the cmake extension and run the auto configure, then click build.

## To Run

For windows,
```cs
Main.exe [config.json]
```

For linux,
```cs
./Main [config.json]
```

For mac OS,
use windows or linux (throw your mac away!)

Adjust hyperparameter in the config.json file. Defaults provided.
Set the dataset file and optional token file. Token file to replace string to int.
Make sure the topology for input and output layer is matching the input and output for the dataset.

## ToDo
1. Export/Import weights
2. Batch Learning
3. Regularization
4. Softmax function for output
5. Normalized input
6. Split into training, validation and test set

