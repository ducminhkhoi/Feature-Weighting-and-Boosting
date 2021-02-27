
# model to run: 0, 2, 3, 5, 8, 9 (for K-shots)
# done: 2, 5

CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_from_scratch --dataset pascal --backbone vgg16 --model 0 --group 0 --iteration 20000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 500 --scratch
CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_from_scratch --dataset pascal --backbone vgg16 --model 0 --group 1 --iteration 20000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 500 --scratch
CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_from_scratch --dataset pascal --backbone vgg16 --model 0 --group 2 --iteration 20000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 500 --scratch
CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_from_scratch --dataset pascal --backbone vgg16 --model 0 --group 3 --iteration 20000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 500 --scratch

# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone vgg16 --model 2 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone resnet101 --model 2 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone resnet101 --model 2 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone resnet101 --model 2 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone resnet101 --model 2 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --train --exp few_shot_official --dataset coco --backbone resnet101 --model 0 --group all --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official --dataset coco --backbone vgg16 --model 0 --group all --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_resnet --dataset coco --backbone resnet101 --model 2 --group all --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200


# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone vgg16 --model 2 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official_sub --dataset pascal --backbone vgg16 --model 2 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000

# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official --dataset coco --backbone vgg16 --model 0 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official --dataset coco --backbone vgg16 --model 0 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official --dataset coco --backbone vgg16 --model 0 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official --dataset coco --backbone vgg16 --model 0 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 1000
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_resnet --dataset coco --backbone resnet101 --model 2 --group all --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=0 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1 python main_official.py --train --exp few_shot_official_vgg --backbone vgg16 --model 4 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200