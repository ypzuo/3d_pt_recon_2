python train_lm_img2ae.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/6/lm_img2ae_lamp_emd \
	--gpu 0 \
	--ae_logs expts/6/pure_ae_1024_lamp_cd \
	--category lamp \
	--bottleneck 512 \
	--loss emd \
	--batch_size 128 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 40 \
	--print_n 100
	# --sanity_check
