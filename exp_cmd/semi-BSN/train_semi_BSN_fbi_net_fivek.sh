#! /bin/bash
cd ../
# initial setting
GPU_NUM=0
#DATE=`date "+%y%m%d"`
DATE="230202" #"220907"

# 
SHORT=d:,g:,s:,a:,b:,t:,p:,h
LONG=date:,gpu:,seed:,alpha:,beta,BSN_type,BSN_param:,help
help()
{
    echo "Usage: $0 [-d][--date] [-g|--gpu GPU_NUM] [-a|--alpha ALPHA] [-b|--beta BETA] [-t|--BSN_type BSN_TYPE] [-p|--BSN_param BSN_PARAM] [-h|--help]"
    echo "  -d, --date DATE: DATE"
    echo "  -g, --gpu GPU_NUM: GPU number"
    echo "  -s, --seed SEED: SEED"
    echo "  -a, --alpha ALPHA: ALPHA"
    echo "  -b, --beta BETA: BETA"
    echo "  -t, --BSN_type BSN_TYPE: BSN_TYPE ['normal-BSN','slightly-BSN','prob-BSN']"
    echo "  -p, --BSN_param BSN_PARAM: BSN_PARAM"
    echo "  -h, --help: print help"
    exit 0
}
OPTS=$(getopt -a -n train_sem --options $SHORT --longoptions $LONG -- "$@" )
eval set -- "$OPTS"
while :
do
    case "$1"  in
        -d | --date )
            DATE="$2"
            shift 2
            ;;
        -g | --gpu )
            GPU_NUM="$2"
            shift 2
            ;;
        -s | --seed )
            SEED="$2"
            shift 2
            ;;
        -a | --alpha )
            ALPHA="$2"
            shift 2
            ;;
        -b | --beta )
            BETA="$2"
            shift 2
            ;;
        -t | --BSN_type )
            BSN_TYPE="$2"
            shift 2
            ;;
        -p | --BSN_param )
            BSN_PARAM="$2"
            shift 2
            ;;
        -h | --help)
            echo "This is a test semi-BSN script"
            help
            exit 2
            ;;
        --)
            break
            ;;
        *)
            echo "Unexpected option: $1"
            help
            ;;
    esac
done

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

# ALPHA == 0, BETA == 0 : Mixture Noise
# ALPHA=$1
# BETA=$2 # == sigma of Poisson Gaussian noise
echo "DATE : ${DATE} GPU_NUM :"${GPU_NUM}
echo "ALPHA :"${ALPHA}" BETA : "${BETA}
echo "BSN_TYPE :"${BSN_TYPE}" BSN_PARAM : "${BSN_PARAM}" SEED : "${SEED}
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date "220907" --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 220
#echo $ALPHA $BETA
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE\
    --seed $SEED --nepochs 20 \
    --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --BSN-type $BSN_TYPE --BSN-param $BSN_PARAM \
    --data-type 'RawRGB' --data-name 'fivek' \
    --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 \
    --num-layers 17 --num-filters 64 --crop-size 200 \
    --pge-weight-dir '211127_PGE_Net_RawRGB_fivek_alpha_'$ALPHA'_beta_'$BETA'_cropsize_200.w' 
