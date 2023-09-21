#pragma once 

#include "config.h"


#pragma warning( push, 0 )

#include <torch/torch.h>

#pragma warning( pop ) 




struct NoveltyEncoder : torch::nn::Module {

    NoveltyEncoder(int interfaceSize)
    {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(interfaceSize, 32)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(32, 16)));
        fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(16, 5)));
        fc4 = register_module("fc4", torch::nn::Linear(torch::nn::LinearOptions(5, 16)));
        fc5 = register_module("fc5", torch::nn::Linear(torch::nn::LinearOptions(16, 32)));
        fc6 = register_module("fc6", torch::nn::Linear(torch::nn::LinearOptions(32, interfaceSize)));

        //torch::NoGradGuard no_grad;
        //for (auto& p : named_parameters()) {
        //    std::string y = p.key();
        //    auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

        //    if (y.compare(2, 6, "weight") == 0) {
        //        torch::nn::init::kaiming_normal_(z);
        //        //torch::nn::init::xavier_normal_(z);
        //    }
        //}
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        x = torch::tanh(fc4->forward(x));
        x = torch::tanh(fc5->forward(x));
        x = torch::tanh(fc6->forward(x));
        return x;
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr }, fc6{ nullptr };
};
