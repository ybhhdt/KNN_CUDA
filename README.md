# KNN_CUDA

+ ref: [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA)
+ ref: [pytorch knn cuda](https://github.com/chrischoy/pytorch_knn_cuda)
+ ref: [kNN-CUDA](https://github.com/unlimblue/KNN_CUDA)


#### Modifications 
+ Added a `threshold` feature to remove overlapping points
+ Aten support
+ pytorch v1.0+ support
+ pytorch c++ extention 

#### Performance

+ dim   = 5
+ k     = 100
+ ref   = 224
+ query = 224
+ Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
+ NVIDIA GeForce 940MX

| Loop   | sklearn | CUDA    | Memory   |
| :---:  | :---:   | :---:   | :---:    |
| 100    | 2.34 ms | 0.06 ms | 652/1024 |
| 1000   | 2.30 ms | 1.40 ms | 652/1024 |


#### Install


+ from source

```bash
git clone https://github.com/ybhhdt/KNN_CUDA.git
cd KNN_CUDA
make && make install
```

+ from wheel

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
And then, make sure [`ninja`](https://ninja-build.org/) has been installed:
  1. see [https://pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html)
  2. **or just**:
```bash
wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
```

+ for windows

You should use branch `windows`:

```bash
git clone --branch windows https://github.com/unlimblue/KNN_CUDA.git
cd C:\\PATH_TO_KNN_CUDA
make
make install
```

#### Usage

```python
import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

"""
When the distance between two points is less than 1e-8,
their coordinates can be considered identical,
and the overlapping points will be excluded from the KNN consideration.
"""
knn = KNN(k=10, transpose_mode=True,threshold=True)
#knn = KNN(k=10, transpose_mode=True)#The default value of the threshold is set to False
ref = torch.rand(32, 1000, 5).cuda()
query = torch.rand(32, 50, 5).cuda()
dist, indx = knn(ref, query)  # 32 x 50 x 10

```
