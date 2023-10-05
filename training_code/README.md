# Relaxed Noise2Noise (RN2N) with FBI-denoiser
- The original code of FBI-denoiser is from [here](https://github.com/csm9493/FBI-Denoiser)

## Folder information
- core
  - 학습 & inference 코드가 있는 폴더
- data_preparation
  - 큰 이미지 (2048x3072)를 학습에 편하게 patch (256x256)로 쪼개서 hdf5 파일로 저장하는 코드가 있는 폴더
- dataset
  - training/test에 사용할 이미지두는 폴더 (학습에 필요한 이미지를 두면 됩니다.)
  - ![Alt text](dataset_example.png)
  - dataset_example.png 또는 아래를 참고하여, dataset 폴더내부를 구성하시면 됩니다.
    - dataset
      - F#1
        - F#1_01.png
        - F#1_02.png
        - ...
      - F#2
        - F#2_01.png
        - F#2_02.png
        - ...
  - dataset 내부에 F01,F32 이미지가 아닌 다른 이미지를 넣으시면, F01,F32 폴더를 삭제하시고, 해당하는 F# 폴더를 만드시면 됩니다.
    - ex) F01-F02 학습 -> F01,F02 폴더 생섯
  - 해당 이미지 파일이름 형식(F#_xx.png)으로 작성해주셔야 코드수정없이 코드를 돌릴 수 있습니다.
- get_denoised_result
  - 학습한 모델의 weight를 활용해, 이미지 디노이징을 수행하는 "get_denoised_output.ipynb" 파일이 있는 폴더.
- result_data
  - 학습과정에서의 결과(PSNR,SSIM)와 이미지가 저장되는 파일이 있는 곳 (scipy.io 로 열 수 있습니다.)
- samsung_log
  - tensorboard 로그가 저장되는 곳 (wandb를 사용하시면 필요없습니다.)
- wandb
  - wandb 로그가 저장되는 폴더 
- weights
  - 학습된 모델의 weight가 저장되는 폴더 
  - 해당 폴더에 있는 weight가 get_denoised_output.ipynb에서 쓰입니다.

## training guideline
1. data_preparation 
   - 학습시 사용할 patch를 만드는 과정입니다.
   1. dataset 폴더에 학습하고자하는 f_number 폴더를 만들고, 그안에 해당하는 이미지를 넣어줍니다.
      - F01,F32를 이용해 학습하고자하신다면, (현재 만들어져있는) F01,F32 폴더를 활용하시고, 아니라면 해당 폴더를 지우시고 해당하는 f_number를 만드시면 됩니다.
      - 폴더안에 F01_xx.png 양식으로 넣어줍니다. (해당 양식을 사용하시지 않으려면, 코드를 수정하셔야합니다.)
      - 두 폴더안에 이미지 개수는 동일해야 합니다. (image-image pair로 사용하기 위해)
   2. data_preparation 폴더안에 있는 run_make_patch.sh 실행하시면 됩니다.
      - make_patch.py 관련 옵션은 make_patch.py 파일 line 15~20에 있습니다.
2. training
   1. Samsung_RN2N.sh 코드 실행
      - ex) ./Samsung_RN2N.sh 0 F01 F32 : 0번 GPU에서 RN2N F01-F32 학습 
      - 기본 세팅은 epoch 10으로 되어있습니다. loss curve를 기준으로 알맞게 수정하시면 될 것같습니다.
      - l1 norm on image gradient 활용하시려면, Samsung_RN2N.sh 코드에서, 
        - line 10의 LOSS_FUNCTION='MSE' -> LOSS_FUNCTION='MSE_with_l1norm_on_gradient'
        - line 31 뒤에 --lambda_val 0.005 를 추가하셔서 lambda 값을 수정하시면 될 것같습니다. (default는 0.005)
3. denoise with trained weight
   1. get_denoised_output.ipynb 파일을 실행하여 디노이징 결과를 얻으시면 됩니다.
      - 학습이후 weights 폴더안에 ~.w 파일이 하나있다면, 그냥 돌리시면 됩니다.
        - 아니라면, 여러개 폴더가 만들어질텐데 맞는 모델의 폴더를 확인하시면 됩니다.
      - 모델에 해당하는 폴더가 만들어지며 해당 모델로 디노이징한 결과가 해당 폴더안에 저장됩니다. 
        - 전체 크기 기준으로 디노이징 o, patch 단위로 디노이징 결과 x