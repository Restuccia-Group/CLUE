#!/bin/bash

# Declare the arrays
unlearn=("bdist")
class_to_replace=(2)
unlearn_epochs=(5)
unlearn_lr=(0.01) #Cifar10
# unlearn_lr=(0.01 0.02 0.05) #Cifar100
#threshold=(0.01 0.05 0.1 0.5) #Cifar100

#Additional for cifar10
threshold=(0.003)
#threshold=(0.1 0.15 0.2)

#Additional for cifar100 vit_b
#unlearn_lr=(0.005 0.02 0.03)
#threshold=(0.01 0.02 0.03)

#unlearn_lr=(0.001 0.0001 0.00001)
#threshold=(0.05 0.1 0.2)



# Iterate over each combination
for ul in "${unlearn[@]}"; do
  for cr in "${class_to_replace[@]}"; do
    for ue in "${unlearn_epochs[@]}"; do
      for lr in "${unlearn_lr[@]}"; do
        for th in "${threshold[@]}"; do

          # Print or execute your command here
          echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"
          
          # python main_forget.py --batch_size 256 --weight_decay=0 --retain_percentage=1.0 --dataset cifar100 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_model_SA_best.pth.tar' --unlearn $ul --class_to_replace $cr --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
          python main_forget.py --batch_size 256 --weight_decay=0 --retain_percentage=1.0 --dataset cifar100 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_model_SA_best.pth.tar' --unlearn $ul --num_indexes_to_replace 4500 --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
        done
      done
    done
  done
done


# Declare the arrays
# unlearn=("ood")
# class_to_replace=(0 12)
# #class_to_replace=(2)
# unlearn_epochs=(8)
# unlearn_lr=(0.00001) #Cifar10
# #unlearn_lr=(0.01 0.02 0.05) #Cifar100
# #threshold=(0.01 0.05 0.1 0.5) #Cifar100

# #Additional for cifar10
# threshold=(0.1)
# #threshold=(0.1 0.15 0.2)

# #Additional for cifar100 vit_b
# #unlearn_lr=(0.005 0.02 0.03)
# #threshold=(0.01 0.02 0.03)

# #unlearn_lr=(0.001 0.0001 0.00001)
# #threshold=(0.05 0.1 0.2)



# # Iterate over each combination
# for ul in "${unlearn[@]}"; do
#   for cr in "${class_to_replace[@]}"; do
#     for ue in "${unlearn_epochs[@]}"; do
#       for lr in "${unlearn_lr[@]}"; do
#         for th in "${threshold[@]}"; do

#           # Print or execute your command here
#           echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"
          
#           python main_forget.py --batch_size 256 --weight_decay=0 --retain_percentage=1.0 --dataset cifar100 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/Resnet20s_cifar100_model_SA_best.pth.tar' --unlearn $ul --class_to_replace $cr --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch resnet20s
#         done
#       done
#     done
#   done
# done