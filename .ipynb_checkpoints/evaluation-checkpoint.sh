# PRCC
python main.py --gpu_devices 0 --dataset prcc --dataset_root data --dataset_filename prcc --resume log/Baseline_2/eps_0.6/prcc/best_mAP_model.pth --save_dir log/Baseline_2/eps_0.6/prcc --evaluate
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root data --dataset_filename LTCC_ReID --resume log/Baseline_2/eps_0.6/ltcc/best_mAP_model.pth --save_dir log/Baseline_2/eps_0.6/ltcc --evaluate