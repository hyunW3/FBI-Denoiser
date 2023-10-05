#! /bin/bash
cd ../

SHORT=d:,g:,v:,tr:,h
LONG=date:,gpu:,dataset-version:,train-set:,help
help()
{
    echo "Usage: $0 [-d][--date] [-g|--gpu GPU_NUM] [--ver|--dataset-version v1 or v2] [-- TRAIN=SET] [-h|--help]"
    echo "  -d, --date DATE: DATE"
    echo "  -g, --gpu GPU_NUM: GPU number"
    echo "  -v --dataset-version: dataset version"
    echo "  --              : TRAIN_SET"
    echo "  -h, --help: print help"
    exit 0
}
OPTS=$(getopt -a -n train_sem --options $SHORT --longoptions $LONG -- "$@" )
echo $OPTS

eval set -- "$OPTS"
echo $OPTARG
VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 1 ]; then
  help
fi
while :
do
  case "$1" in
    -d | --date )
      DATE="$2"
      shift 2
      ;;
    -g | --gpu )
      GPU_NUM="$2"
      shift 2
      ;;
    -v | --dataset-version )
      DATASET_VERSION="$2"
      shift 2
      ;;
    -h | --help)
      echo "This is a test different trainset script"
      help
      exit 2
      ;;
    --)
      TRAIN_SET="${@:2}"
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
done

# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
echo "DATE : ${DATE} Integrated SET GPU_NUM :"${GPU_NUM}
echo "DATASET-VER :"${DATASET_VERSION}" TRAIN_SET : "${TRAIN_SET}

BATCH_SIZE=1

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation

echo "=== Train FBI with MSE_AFFINE === integrated SET"
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --nepochs 30 \
    --integrate-all-set --test-wholedataset \
    --train-set $TRAIN_SET --wholedataset-version $DATASET_VERSION \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' \
    --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 \
    --num-filters 64 --crop-size 256 
