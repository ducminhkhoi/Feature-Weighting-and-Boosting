

CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official --backbone resnet101 --dataset coco --num_shots 1 --model 5 --group 0 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone vgg16 --dataset pascal --num_shots 1 --model 2 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone vgg16 --dataset pascal --num_shots 1 --model 2 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone vgg16 --dataset pascal --num_shots 1 --model 2 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8

# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone resnet101 --dataset pascal --num_shots 1 --model 2 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone resnet101 --dataset pascal --num_shots 1 --model 2 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone resnet101 --dataset pascal --num_shots 1 --model 2 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python main_official.py --exp few_shot_official_sub --backbone resnet101 --dataset pascal --num_shots 1 --model 2 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8

# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official --backbone resnet101 --dataset pascal --num_shots 5 --model 9 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset pascal --num_shots 1 --model 5 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset pascal --num_shots 1 --model 5 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset pascal --num_shots 1 --model 5 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200


# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset coco --num_shots 5 --model 8 --group 0 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset coco --num_shots 5 --model 8 --group 1 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset coco --num_shots 5 --model 8 --group 2 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200
# CUDA_VISIBLE_DEVICES=1,0 python main_official.py --exp few_shot_official_vgg --backbone vgg16 --dataset coco --num_shots 5 --model 8 --group 3 --iteration 10000 --base_lr 0.007 --num_folds 4 --crop_size 512 --batch_size 8 --val_interval 200