#pragma once

// only included by moevCore.h for simplicity, since all files must include moevCore.h.

// https://docs.huihoo.com/eigen/3/TopicMultiThreading.html
#define EIGEN_DONT_PARALLELIZE
#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Core>
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MMatrix; // mapped matrix
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> MVector; // column mapped vector

#define WRITE_4B(i, os) os.write(reinterpret_cast<const char*>(&i), 4);
#define READ_4B(i, is) is.read(reinterpret_cast<char*>(&i), 4);
#define LOGV(v) for (const auto e : v) {std::cout << std::setprecision(2)<< e << " ";}; std::cout << "\n"
#define LOG(x) std::cout << x << std::endl;

