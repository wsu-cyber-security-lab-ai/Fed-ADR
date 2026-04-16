#!/bin/bash
# declare -a AR_METHOD=("mudhog" "fedavg" "median" "gm" "krum" "mkrum" "foolsgold")
# declare -a AR_METHOD=("mudhog" "krum" "trimmed" "fedavg" "median" "gm" "mkrum" "foolsgold")
declare -a AR_METHOD=("fedavg" "median" "gm" "foolsgold" "krum" "trimmed")

#### LIST CLIENTS for MNIST dataset
# Unrelibale: upto 4 nodes
# LIST_UNRELIABLE=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37")

# Untargeted: Sign-flipping attackers up to 5 nodes
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23" )

# Untargeted: Additive noise attackers, upto 6 node
# LIST_AN_ATTACKERS=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28,30")

# Targeted: Label-flipping attackers: increasing from 3 nodes to 8 nodes.
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

# Targeted: Multi-labeli-flipping attackers: increasing from 3 nodes to 8 nodes.
# LIST_MLF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38" )
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "39,40,41,42,43")
# LIST_MLF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "44,45,46,47,48,49,50,51,52,53")



# LIST_SF_ATTACKERS=(
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
#         "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"
#          "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
#             )
# DATASET="mnist"
# LR=1e-2 # SGD optimizer learning rate
# WD=1e-4  # weight-decay
# MOMENTUM=0.5 # SGD optimizer momentum
# GLOBAL_EPOCHS=40
# LOCAL_EPOCHS=4
# STD_UR=30
# NUM_CLIENTS=100
# MINIMUM_ITERATION_TO_START_THE_ATTACK=15
# LOADER_TYPE=dirichlet

# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18" )
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "19,20,21,22,24,25,26,27,28")



# We train the FL model with SGD optimizer (learning
# rate = 1e-2, momentum = 0.5 for MNIST and 0.9 for Fashion-MNIST, and
# weight-decay = 1e-4 for Fashion-MNIST) over 40 communication rounds, 4 local
# epochs;

# Series Exp1A (AN_LF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (LF) attackers; where j in range [2,6].
#EXP_NAME="sExp1Bb_40ep" 
#for ((i=0; i < 1; i++));
#do
#for ((j=4; j<6; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --max_std_unreliable $STD_UR &
#done
#done

# sExp2A (AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) multi-label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp2Ba_40ep"
#for ((i=1; i < 2; i++));
#do
#for ((j=2; j<4; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients 40  --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --source_labels 1,2,3
#done
#done


# Series Exp3A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp3Bb_40ep"
#for ((i=0; i < 1; i++));
#do
#for ((j=2; j<3; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS  --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR 
#done
#done

# Series Exp4A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+2) multi-label-flipping (MLF) attackers; where j in range [1,5].
# EXP_NAME="sExp4Ba_40ep10"



# Untargeted: Sign-flipping attackers up to 5 nodes
# LIST_SF_ATTACKERS=("0,10" "0,10,11,18" "0,10,11,18,19,20" "0,10,11,18,19,20,24,28" "0,10,11,18,19,20,24,28,29,30" "0,10,11,18,19,20,24,28,29,30")
# LIST_SF_ATTACKERS=("0,10" "0,10,11,18" "0,10,11,18,19,20" "0,10,11,18,19,20,24,28" "0,10,11,18,19,20,24,28,29,30" "0,10,11,18,19,20,24")
# LIST_SF_ATTACKERS=("0,10" "0,10,11,18" "0,10,11,18,19,20" "0,10,11,18,19,20,24,28" "0,10,11,18,19,20,24,28,29,30" "0,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20,24,28,29")



# LIST_SF_ATTACKERS=("0,10,11,12,13" "0,10,11,12,13,14,15,16" "0,10,11,12,13,14,15,16,17,18,19" "0,10,11,12,13,14,15,16,17,18,19,20,21,22" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")

# DATASET="mnist"
# LR=1e-2 # SGD optimizer learning rate
# WD=1e-4  # weight-decay
# MOMENTUM=0.5 # SGD optimizer momentum
# GLOBAL_EPOCHS=200
# LOCAL_EPOCHS=4
# STD_UR=30
# NUM_CLIENTS=40
# MINIMUM_ITERATION_TO_START_THE_ATTACK=50
# LOADER_TYPE=dirichlet


# LIST_SF_ATTACKERS=(
    # "0,10"      #5
    # "0,10,11,12"      #10
    # "0,10,11,12,13,14"      #15
    # "0,10,11,12,13,14,15,16"      #20
    # "0,10,11,12,13,14,15,16,17,18"      #25
    # "0,10,11,12,13,14,15,16,17,18,19,20"      #30
    # "0,10,11,12,13,14,15,16,17,18,19,20,21,22"      #35
    # "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"      #40
    # "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"      #45
# )
# LIST_LF_ATTACKERS=(
#     "27,28,29,30"
#     # "27,28,29,30"
#     "27,28,29,30"
#     # "27,28,29,30"
#     "27,28,29,30"
#     # "27,28,29,30"
#     "27,28,29,30"
#     # "27,28,29,30"
#     # "27,28,29,30"
# )

# LIST_SF_ATTACKERS=(
#     "0,10" 
#     "0,10,11,12,13,14"
#     "0,10,11,12,13,14,15,16"
#     "0,10,11,12,13,14,15,16,17,18"
# )
# LIST_LF_ATTACKERS=(
#     "27,28"
#     "27,28"
#     "27,28,29"
#     "27,28,29,30"
# )
# LIST_MLF_ATTACKERS=(
#     "31,32"
#     "31,32"
#     "31,32,33"
#     "31,32,33,34"
# )


# # 100 mix
# LIST_SF_ATTACKERS=(
#     "4,12,21,30,35,38,48"
#     "3,11,16,19,26,30,35,37,40,42,45,47,49,52"
#     "3,4,8,9,12,13,14,15,16,29,31,33,37,45,47,48,53,55,59,60,64"
#     "0,1,2,3,11,14,16,18,20,22,23,27,30,31,35,36,41,43,48,49,52,53,58,59,60,61,70,73"
#     "2,3,4,6,8,9,12,15,18,19,22,23,24,25,29,30,34,36,39,40,41,42,44,46,51,52,56,57,58,61,63,64,65,68,74"
# )

# LIST_LF_ATTACKERS=(
#     "55,67,89"
#     "55,60,64,66,81,91"
#     "65,69,72,76,78,82,88,93,95"
#     "74,77,79,80,81,82,84,87,92,93,94,95"
#     "77,78,79,81,82,83,85,87,88,93,94,95,96,97,99"
# )

# 100
LIST_SF_ATTACKERS=(
    "4,12,21,30,35,38,48,55,67,89"        #10
    "3,11,16,19,26,30,35,37,40,42,45,47,49,52,55,60,64,66,81,91"        #20
    "3,4,8,9,12,13,14,15,16,29,31,33,37,45,47,48,53,55,59,60,64,65,69,72,76,78,82,88,93,95"        #30
    # "0,1,2,3,11,14,16,18,20,22,23,27,30,31,35,36,41,43,48,49,52,53,58,59,60,61,70,73,74,77,79,80,81,82,84,87,92,93,94,95"        #40
    # "2,3,4,6,8,9,12,15,18,19,22,23,24,25,29,30,34,36,39,40,41,42,44,46,51,52,56,57,58,61,63,64,65,68,74,77,78,79,81,82,83,85,87,88,93,94,95,96,97,99"        #50
)

# 200
# LIST_SF_ATTACKERS=(
#     "35,41,42,56,64,99,103,109,118,120,128,133,143,144"
#     "0,1,2,6,22,25,32,38,44,46,50,52,53,57,59,65,70,75,80,83,88,101,104,105,106,107,112,118"
#     "2,5,6,10,12,13,14,18,20,24,26,27,28,31,40,42,44,45,56,57,60,69,70,76,81,92,93,94,97,100,106,112,113,115,116,121,126,129,136,140,141,144"
#     "0,3,6,8,10,11,13,15,16,22,24,32,38,42,43,50,51,61,62,64,66,67,69,73,74,75,77,80,81,82,87,88,91,93,96,100,101,102,103,105,108,118,120,125,131,133,134,135,136,137,139,141,142,143,145,147"
#     "2,3,5,6,8,11,14,16,17,21,23,25,26,27,28,29,32,41,43,45,47,50,52,54,57,58,59,61,65,67,69,70,72,74,77,78,79,80,81,82,83,84,85,86,89,91,92,94,96,98,99,101,103,104,105,107,108,111,114,115,117,118,120,121,122,123,124,126,127,130"
# )

# LIST_LF_ATTACKERS=(
#     "149,156,179,187,194,195"
#     "131,134,152,153,162,168,176,177,187,188,190,197"
#     "146,152,154,156,158,162,165,170,173,174,175,178,179,187,189,190,198,199"
#     "148,152,153,157,160,162,164,167,172,174,175,176,177,180,182,184,185,186,188,189,192,193,196,197"
#     "136,137,139,140,142,145,147,150,152,153,157,160,161,163,164,166,168,172,174,176,177,184,186,187,190,193,194,196,197,198"
# )


# 300
# LIST_SF_ATTACKERS=(
#     "2,3,9,11,25,28,52,54,75,87,137,140,141,143,144,149,172,183,188,192,193"
#     "3,27,29,31,46,50,52,61,62,71,78,88,94,98,100,104,116,126,137,140,141,145,147,149,159,162,165,168,174,178,180,184,186,192,198,199,201,209,223,224,232,239"
#     "1,3,4,5,12,14,18,19,21,26,29,30,32,34,35,38,40,50,52,54,57,58,61,66,67,68,73,74,83,86,91,93,98,100,104,106,110,113,127,128,133,134,135,136,142,144,145,147,150,153,154,155,156,157,159,161,164,165,166,170,176,177"
#     "0,2,6,11,12,13,14,15,18,19,20,21,23,27,31,34,35,36,38,40,45,46,47,50,51,54,61,63,64,65,66,68,73,74,79,80,81,83,88,90,100,102,106,109,114,115,117,118,124,126,128,132,134,137,138,139,142,143,144,145,146,147,149,158,162,163,164,167,168,170,173,174,178,180,183,184,185,186,187,190,195,196,197,200"
#     "0,1,2,6,8,11,13,15,17,18,21,24,26,27,29,30,31,33,35,37,38,41,43,45,48,50,53,57,58,60,61,62,63,66,68,71,72,75,76,79,80,81,83,84,87,90,91,93,94,99,100,101,104,107,109,111,116,117,118,119,120,121,124,126,127,130,131,132,135,136,138,141,144,145,147,148,149,152,158,159,160,164,165,167,171,172,174,175,176,177,180,181,184,186,187,188,189,191,192,193,194,195,196,197,198"
# )

# LIST_LF_ATTACKERS=(
#     "194,208,211,258,259,261,269,270,284"
#     "244,247,248,250,253,258,262,266,267,271,272,274,282,283,285,293,294,296"
#     "180,181,186,191,193,199,206,209,211,221,226,232,233,241,247,248,259,261,264,265,267,268,272,289,292,293,296,297"
#     "201,203,209,212,213,217,219,221,223,225,228,229,232,238,244,250,251,253,255,256,261,262,263,268,269,276,279,282,284,285,286,292,293,294,296,297"
#     "199,201,204,205,207,211,215,218,219,220,221,222,223,226,227,228,229,232,236,237,243,245,247,251,252,253,254,255,260,261,269,270,273,276,279,280,282,283,288,289,290,294,295,298,299"
# )

# DATASET="mnist"
# LR=1e-2 # SGD optimizer learning rate
# WD=1e-4  # weight-decay
# MOMENTUM=0.5 # SGD optimizer momentum
# GLOBAL_EPOCHS=40
# LOCAL_EPOCHS=4
# STD_UR=30
# NUM_CLIENTS=40
# MINIMUM_ITERATION_TO_START_THE_ATTACK=10
# LOADER_TYPE=dirichlet

DATASET="fashion_mnist"
LR=1e-2 # SGD optimizer learning rate
WD=1e-4  # weight-decay
MOMENTUM=0.9 # SGD optimizer momentum
GLOBAL_EPOCHS=40
LOCAL_EPOCHS=4
STD_UR=30
NUM_CLIENTS=100
MINIMUM_ITERATION_TO_START_THE_ATTACK=10
LOADER_TYPE=dirichlet

# DATASET="cifar"
# LR=1e-2 # SGD optimizer learning rate
# WD=1e-4  # weight-decay
# MOMENTUM=0.9 # SGD optimizer momentum
# GLOBAL_EPOCHS=40
# LOCAL_EPOCHS=4
# STD_UR=30
# NUM_CLIENTS=40
# MINIMUM_ITERATION_TO_START_THE_ATTACK=0
# LOADER_TYPE=dirichlet


for ((i=5; i < 6; i++));
# for ((i=1; i < 2; i++));
# for ((i=0; i < 1; i++));
do
for ((j=2; j<3; j++));
do
# STARTING_THRESHOLD=0
EXP_NAME="different_aggregation_methods2_${NUM_CLIENTS}C_40ep_${MINIMUM_ITERATION_TO_START_THE_ATTACK}start_${j}"
# EXP_NAME="gauassian_label_multi_show_testing_clients_Exp_${NUM_CLIENTS}C_40ep_${MINIMUM_ITERATION_TO_START_THE_ATTACK}start_${j}"
# EXP_NAME="Gassuian_only_different_aggregation_methods_comparison_Exp_40C_40ep_${MINIMUM_ITERATION_TO_START_THE_ATTACK}start_${j}"
# EXP_NAME="Gassuian_and_label_flipping_and_multi_lebel_Exp_40C_40ep_${MINIMUM_ITERATION_TO_START_THE_ATTACK}start_${j}"
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --attacks "backdoor" --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_backdoor ${LIST_BD_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 

CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]} --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]} --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]} --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]} --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 

done
done
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK & 
#  sudo pkill -f main.py