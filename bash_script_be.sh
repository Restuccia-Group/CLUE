#!/bin/bash

# Declare the arrays
unlearn=("boundary_shrink")
class_to_replace=(2)
#class_to_replace=(2)
unlearn_epochs=(10)
unlearn_lr=(0.00001)
threshold=(0.02)

# Iterate over each combination
# for ul in "${unlearn[@]}"; do
#   for cr in "${class_to_replace[@]}"; do
#     for ue in "${unlearn_epochs[@]}"; do
#       for lr in "${unlearn_lr[@]}"; do
#         for th in "${threshold[@]}"; do

#           # Print or execute your command here
#           echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"
          
#           python main_forget.py --batch_size 128 --weight_decay=0 --retain_percentage=1.0 --dataset cifar100 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_model_SA_best.pth.tar' --unlearn $ul --class_to_replace $cr --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
#         done
#       done
#     done
#   done
# done


# Iterate over each combination
for ul in "${unlearn[@]}"; do
  for ue in "${unlearn_epochs[@]}"; do
    for lr in "${unlearn_lr[@]}"; do
      for th in "${threshold[@]}"; do

        # Print or execute your command here
        echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"
        
        python main_forget.py --batch_size 32 --retain_percentage=1.0 --weight_decay=0 --dataset cifar100 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_model_SA_best.pth.tar' --unlearn $ul --class_to_replace 2 --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
      done
    done
  done
done

#Cifar100
# Declare the arrays
# unlearn=("boundary_expanding")
# class_to_replace=(0 4 6 8)
# #class_to_replace=(2)
# unlearn_epochs=(10)
# unlearn_lr=(0.0001)
# threshold=(0.02)

# # Iterate over each combination
# for ul in "${unlearn[@]}"; do
#   for cr in "${class_to_replace[@]}"; do
#     for ue in "${unlearn_epochs[@]}"; do
#       for lr in "${unlearn_lr[@]}"; do
#         for th in "${threshold[@]}"; do

#           # Print or execute your command here
#           echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"
          
#           python main_forget.py --batch_size 128 --weight_decay=0 --retain_percentage=1.0 --dataset cifar10 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar10_model_SA_best.pth.tar' --unlearn $ul --class_to_replace $cr --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
#         done
#       done
#     done
#   done
# done