# 总流程
数据预处理：
    -使用colmap对数据进行稀疏重建，并转换坐标系为enu坐标系，camera和点云信息存储在DATA/sparse/0下
环境:
```bash
# download
git clone https://github.com/hurriedl/hurriedl_test.git --recursive

conda env create --file environment.yml
conda activate surfel_splatting
```

训练：
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

渲染：
```bash
python render.py -m <path to pre-trained model>

#正射图像
python render.py -m <path to pre-trained model> --ortho --image_blending --skip_mesh
```