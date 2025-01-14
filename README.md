# 总流程
数据预处理：
    -使用colmap对数据进行稀疏重建，并转换坐标系为enu坐标系，camera和点云信息存储在DATA/sparse/0下
环境:
```bash
# download
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

conda env create --file environment.yml
conda activate surfel_splatting
```
