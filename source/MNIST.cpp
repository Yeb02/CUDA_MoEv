#pragma once 

#include "MNIST.h"

#include <vector>
#include <iostream>


#pragma warning( push, 0 )

#include <torch/torch.h>

#pragma warning( pop ) 


struct Net : torch::nn::Module {

    Net()
    {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(784, 30)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(30, 10)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::sigmoid(fc1->forward(x));
        x = torch::sigmoid(fc2->forward(x));
        return x;
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};


void testCustomReader() {
    float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", 10000);
    float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", 10000);

    float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", 60000);
    float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", 60000);

    int testLabelsV[10000];
    std::vector<torch::Tensor> tensorTestDatapointsList;
    tensorTestDatapointsList.resize(10000);

    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 10; j++) {
            if (testLabels[i][j] == 1.0f) testLabelsV[i] = j;
        }

        tensorTestDatapointsList[i] = torch::from_blob(testDatapoints[i], { 1, 784 },
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
        );
    }
    torch::Tensor tensorTestDatapoints = torch::stack(tensorTestDatapointsList, 1).squeeze(0);

    const int batchSize = 10;

    torch::Tensor tensorTrainLabels[60000 / batchSize];
    torch::Tensor tensorTrainDatapoints[60000 / batchSize];
    std::vector<torch::Tensor> toBeConcatenated;
    toBeConcatenated.resize(batchSize);

    for (int i = 0; i < 60000 / batchSize; i++) {

        for (int j = 0; j < batchSize; j++) {
            toBeConcatenated[j] = torch::from_blob(trainLabels[i * batchSize + j], { 1, 10 },
                torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
            );
        }
        tensorTrainLabels[i] = torch::stack(toBeConcatenated, 1).squeeze();

        for (int j = 0; j < batchSize; j++) {
            toBeConcatenated[j] = torch::from_blob(trainDatapoints[i * batchSize + j], { 1, 784 },
                torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
            );
        }
        tensorTrainDatapoints[i] = torch::stack(toBeConcatenated, 1).squeeze();

    }

    Net net;

    torch::optim::SGD optimizer(
        net.parameters(), torch::optim::SGDOptions(3.0f / (float)batchSize)
    );


    for (int epoch = 1; epoch < 10; ++epoch) {
        net.train();

        for (int i = 0; i < 60000 / batchSize; i++) {

            net.zero_grad();

            torch::Tensor outputs = net.forward(tensorTrainDatapoints[i]);
            torch::Tensor loss = torch::mse_loss(outputs, tensorTrainLabels[i]);

            loss.backward();

            optimizer.step();

        }

        {
            torch::NoGradGuard noGrad;
            net.eval();

            torch::Tensor outputs = net.forward(tensorTestDatapoints);

            auto [max, idx] = torch::max({ outputs }, 1);


            int s = 0;
            for (int i = 0; i < 10000; i++) {
                s += (idx[i].item<int>() == testLabelsV[i]) ? 1 : 0;
            }

            std::cout << "Epoch " << epoch << " , accuracy  " << (float)s / 10000 << std::endl;
        }
    }
}

void testStandardLoader() {
    const int batchSize = 10;

    auto train_dataset = torch::data::datasets::MNIST(
        "C:\\Users\\alpha\\Bureau\\CUDA_MoEv\\x64\\Debug\\data\\MNIST\\raw"
    ).map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
        (std::move(train_dataset), batchSize);

    auto test_dataset = torch::data::datasets::MNIST(
        "C:\\Users\\alpha\\Bureau\\CUDA_MoEv\\x64\\Debug\\data\\MNIST\\raw",
        torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
        torch::data::make_data_loader(std::move(test_dataset), 10000);

    float* blob = new float[10 * batchSize];

    Net net;

    torch::optim::SGD optimizer(
        net.parameters(), torch::optim::SGDOptions(3.0f / (float)batchSize)
    );


    for (int epoch = 1; epoch < 10; ++epoch) {
        net.train();

        for (const auto& batch : *train_loader) {

            net.zero_grad();


            std::fill(blob, blob + batchSize * 10, 0.0f);
            for (int i = 0; i < batchSize; i++) {
                blob[batchSize * i + batch.target[i].item<int>()] = 1.0f;
            }
            torch::Tensor target = torch::from_blob(blob, { batchSize, 10 },
                torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
            );

            torch::Tensor outputs = net.forward(batch.data.view({ batchSize, 784 }));
            torch::Tensor loss = torch::mse_loss(outputs, target);

            loss.backward();

            optimizer.step();
        }

        {
            torch::NoGradGuard no_grad;
            net.eval();
            int correct = 0;
            for (const auto& batch : *test_loader) { // 1 batch of size 10000

                auto output = net.forward(batch.data.view({ 10000, 784 }));

                auto pred = output.argmax(1);
                correct += pred.eq(batch.target).sum().template item<int64_t>();
            }

            std::cout << "Epoch " << epoch << " , accuracy  " << (float)correct / (float)test_dataset_size << std::endl;
        }
    }
}


float** read_mnist_images(std::string full_path, int number_of_images) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };



    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        int image_size = n_rows * n_cols;

        unsigned char* buffer = new unsigned char[image_size];
        float** dataset = new float* [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            file.read((char*)buffer, image_size);
            dataset[i] = new float[image_size];
            for (int j = 0; j < image_size; j++) {
                dataset[i][j] = static_cast<float>(buffer[j]) / 127.5f - 1.0f;
            }
        }

        delete[] buffer;

        return dataset;
    }
    else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float** read_mnist_labels(std::string full_path, int number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        float** dataset = new float* [number_of_labels];
        char* buffer = new char[number_of_labels];
        file.read(buffer, number_of_labels);
        for (int i = 0; i < number_of_labels; i++) {
            dataset[i] = new float[10];
            std::fill(dataset[i], dataset[i] + 10, .0f);
            //std::fill(dataset[i], dataset[i] + 10, -1.0f);
            dataset[i][buffer[i]] = 1.0f;
        }

        delete[] buffer;
        return dataset;
    }
    else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

