python enhancement_mp.py --test_dir /data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid --enhanced_dir /data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid/noisy_enhanced_sgmse_stage1_mp --ckpt logs/downloaded_ckpt/train_vb_29nqe0uh_epoch=115.ckpt
python calc_metrics.py --test_dir /data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid --enhanced_dir /data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid/noisy_enhanced_sgmse_stage1_mp