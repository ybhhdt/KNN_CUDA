#include <vector>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
//定义一些用于检查张量属性的宏
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) AT_ASSERTM(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == at::Device::Type::CUDA, #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)

// CUDA核心函数声明
void knn_device(
    float* ref_dev, 
    int ref_nb, 
    float* query_dev, 
    int query_nb, 
    int dim, 
    int k, 
    float* dist_dev, 
    long* ind_dev, 
    cudaStream_t stream,
    bool threshold
    );

std::vector<at::Tensor> knn(
    at::Tensor & ref, 
    at::Tensor & query, 
    const int k,
    const bool threshold
    ){
    // 检查输入张量的属性
    CHECK_INPUT(ref, at::kFloat);
    CHECK_INPUT(query, at::kFloat);
    // 获取输入张量的大小和数据指针
    int dim = ref.size(0);//特征的维度，如：x,y,z
    int ref_nb = ref.size(1);//参考点的数量
    int query_nb = query.size(1);//查询点的数量
    float * ref_dev = ref.data_ptr<float>();//参考点张量的数据指针
    float * query_dev = query.data_ptr<float>();//查询点张量的数据指针
    // 创建输出张量，用于存储距离矩阵和最近邻索引矩阵
    auto dist = at::empty({ref_nb, query_nb}, query.options().dtype(at::kFloat));//每个参考点与每个查询点之间的距离
    auto ind = at::empty({k, query_nb}, query.options().dtype(at::kLong));//每个查询点的前 k个最近邻的索引
    float * dist_dev = dist.data_ptr<float>();//获取dist张量的数据指针
    long * ind_dev = ind.data_ptr<long>();
    // 获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用CUDA核心函数进行最近邻计算
    knn_device(
        ref_dev,
        ref_nb,
        query_dev,
        query_nb,
        dim,
        k,
        dist_dev,
        ind_dev,
        stream,
        threshold
    );

    return {dist.slice(0, 0, k), ind};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "KNN cuda version");
}
