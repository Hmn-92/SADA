# PRCC
#python main.py --gpu_devices 0 --dataset prcc --dataset_root data --dataset_filename prcc --max_epoch 50 --save_dir log/Baseline/eps_0.8/prcc --save_checkpoint --eps 0.8
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root data --dataset_filename LTCC_ReID --save_dir log/eps_0.6/ltcc --save_checkpoint --eps 0.6 --max_epoch 80 --fg_start_epoch 20

#python main.py --gpu_devices 0 --dataset prcc --dataset_root data --dataset_filename prcc --max_epoch 80 --save_dir log/eps_0.6/prcc --save_checkpoint --eps 0.6 --fg_start_epoch 20

#改进版本
# python main.py --gpu_devices 0 --dataset ltcc --dataset_root data --dataset_filename LTCC_ReID --save_dir log/Baseline_3/eps_0.6/ltcc --save_checkpoint --eps 0.6 --max_epoch 80 --fg_start_epoch 25 --resume log/Baseline_2/eps_0.6/ltcc/best_mAP_model.pth
