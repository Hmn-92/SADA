# Semantic-driven Augmentation with Dynamic Fine-Grained Optimization (SADF) framework for CC-ReID
Pytorch code for paper "**Semantic-driven Augmentation with Dynamic Fine-Grained Optimization for Cloth-Changing Person Re-Identification**".
(Submitting to The Visual Computer)

Getting Started

### 1.Environment
Python == 3.8
PyTorch == 1.12.1
faiss-gpu == 1.7.2

### 2. Prepare Data
Please download cloth-changing person re-identification datasets and place them in any path DATASET_ROOT:

DATASET_ROOT
	└─ LTCC-reID or Celeb-reID or PRCC or LaST
		├── train
		├── query
		├── gallery

- LTCC [1]: The LTCC dataset can be downloaded from this [website](https://naiq.github.io/LTCC_Perosn_ReID.html).

- PRCC [2]: The PRCC dataset can be downloaded from this [website](http://www.isee-ai.cn/%7Eyangqize/clothing.html).

- Celeb-reID [3]: The Celeb-reID dataset can be downloaded from this [website](https://github.com/Huang-3/Celeb-reID?tab=readme-ov-file.html). 

- LaST [4]: The LaST dataset can be downloaded from this [website](https://github.com/shuxjweb/last.html).


### 3. Training


**Train SADA by**

```sh
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root DATASET_ROOT --dataset_filename LTCC-reID --save_dir SAVE_DIR --save_checkpoint

# Celeb-reID
python main.py --gpu_devices 0 --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --num_instances 4 --save_dir SAVE_DIR --save_checkpoint

# PRCC
python main.py --gpu_devices 0,1 --dataset prcc --dataset_root DATASET_ROOT --dataset_filename PRCC --max_epoch 30 --save_dir SAVE_DIR --save_checkpoint

# DeepChange
python main.py --gpu_devices 0,1 --dataset deepchange --dataset_root DATASET_ROOT --dataset_filename DeepChange --train_batch 64 --fg_start_epoch 45 --save_dir SAVE_DIR --save_checkpoint

# LaST
python main.py --gpu_devices 0,1 --dataset last --dataset_root DATASET_ROOT --dataset_filename LaST --train_batch 64 --num_instances 4 --fg_start_epoch 45 --save_dir SAVE_DIR --save_checkpoint
```

`--dataset_root` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR` with the path to save log file and checkpoints


### 4. Testing

```sh
python main.py --gpu_devices 0 --dataset DATASET --dataset_root DATASET_ROOT --dataset_filename DATASET_FILENAME --resume RESUME_PATH --save_dir SAVE_DIR --evaluate
```

`--dataset`: replace `DATASET` with the dataset name

`--dataset_filename`: replace `DATASET_FILENAME` with the folder name of the dataset

`--resume`: replace `RESUME_PATH` with the path of the saved checkpoint

The above three arguments are set corresponding to Training.


### 5. Results

 **Celeb-reID**

| Backbone  | Rank-1 | Rank-5 | mAP  |
| :-------: |:------:|:------:|:----:|
| ResNet-50 |  67.2  |  81.4  | 20.1 |

- **LTCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  45.7  | 20.3 |
| ResNet-50 |    Standard    |  78.5  | 41.6 |

- **PRCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  67.8  | 64.1 |
| ResNet-50 |    Standard    |  100   | 99.8 |

- **LaST**

| Backbone  | Rank-1 | mAP  |
| :-------: |:------:|:----:|
| ResNet-50 |  76.3  | 34.4 |

**\*The results may exhibit fluctuations due to random splitting, and further improvement can be achieved by fine-tuning the hyperparameters.**



Please cite the following paper in your publications if it is helpful:

###  6. References.

[1] X. Qian, W. Wang, L. Zhang, F. Zhu, Y. Fu, T. Xiang, Y.-G. Jiang, X. Xue. Long-Term Cloth-Changing Person Re-identification. In: Proceedings of the Asian Conference on Computer Vision (ACCV), 2020, https://doi.org/10.1007/978-3-030-69535-4_5.

[2] Q. Yang, A. Wu, W.S. Zheng. Person Re-Identification by Contour Sketch Under Moderate Clothing Change. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 43(6): 2029-2046, https://doi.org/10.1109/TPAMI.2019.2960509.

[3] Y. Huang, J. Xu, Q. Wu, Y. Zhong, P. Zhang, Z. Zhang. Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person Re-Identification. IEEE Transactions on Circuits and Systems for Video Technology, 2020, 30(10): 3459-3471, https://doi.org/10.1109/TCSVT.2019.2948093.

[4] X. Shu, X. Wang, X. Zang, S. Zhang, Y. Chen, G. Li, Q. Tian. Large-scale spatio-temporal person re-identification: Algorithms and benchmark. IEEE Transactions on Circuits Systems for Video Technology, 2021, 32(7): 4390-4403, https://doi.org/10.1109/TCSVT.2021.3128214.
