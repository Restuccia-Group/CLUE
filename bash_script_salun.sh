#!/bin/bash

# Declare the arrays
unlearn=("RL")
class_to_replace=(2)
#class_to_replace=(2)
unlearn_epochs=(10)
unlearn_lr=(0.000005 0.00005 0.0005 0.005 0.05) #Cifar10
#unlearn_lr=(0.01 0.02 0.05) #Cifar100
#threshold=(0.01 0.05 0.1 0.5) #Cifar100

#Additional for cifar10
# threshold=(0.003 0.0003 0.0003)
threshold=(0.5)
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

          # Print or execute your command here
          echo "Processing: unlearn=$ul, class_to_replace=$cr, unlearn_epochs=$ue, unlearn_lr=$lr"

          python main_forget.py --unlearn $ul --unlearn_epochs $ue --unlearn_lr $lr --retain_percentage=1.0 --class_to_replace $cr --dataset cifar10 --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/Resnet20s_cifar10_model_SA_best.pth.tar' --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --mask_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/sal_map/with_0.5.pt' --chenyaofo 0 --arch resnet20s
          
          #python main_forget.py --batch_size 256 --weight_decay=0 --retain_percentage=1.0 --dataset cifar10 --mask_threshold $th --save_dir '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results' --model_path '/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar10_model_SA_best.pth.tar' --unlearn $ul --class_to_replace $cr --unlearn_epochs $ue --unlearn_lr $lr --chenyaofo 0 --arch vit_b
      done
    done
  done
done