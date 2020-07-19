#!/bin/bash

gpu=0
exp=trained_models/lm
dataset=shapenet

eval_set=valid
txt=resu
image=plane4.png


python show_results.py \
	--gpu $gpu \
	--exp $exp \
	--mode lm \
	--load_best \
	--bottleneck 512 \
	--bn_decoder \
	--eval_set ${eval_set}\
	--image ${image}\
	--txt ${txt} \
	--visualize  
