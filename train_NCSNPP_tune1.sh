TORCH_DISTRIBUTED_DEBUG="DETAIL" CUDA_VISIBLE_DEVICES=1 python train.py --base_dir /data2/zhounan/data/noisy/voicebank_demand/sgmse_data --accelerator gpu --devices 1 --theta 5.0 --sigma-min 0.0005