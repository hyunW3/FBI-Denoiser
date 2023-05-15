import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Denoising')
    # Arguments
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--noise-type', default='Real', type=str, required=False,
                        choices=['Poisson-Gaussian'],
                        help='(default=%(default)s)')
    parser.add_argument('--loss-function', default='Estimated_Affine', type=str, required=False,
                        choices=['MSE', 'N2V', 'MSE_Affine', 'Noise_est', 'EMSE_Affine', 'MSE_Affine_with_tv'],
                        help='(default=%(default)s)')
    parser.add_argument('--lambda-val', default=5, type=float, help='(default=%(default)f)')
    parser.add_argument('--vst-version', default='MSE', type=str, required=False,
                        choices=['MSE', 'MAE'])
    parser.add_argument('--model-type', default='final', type=str, required=False,
                        choices=['case1',
                                 'case2',
                                 'case3',
                                 'case4',
                                 'case5',
                                 'case6',
                                 'case7',
                                 'FBI_Net',
                                 'PGE_Net',
                                 'DBSN',
                                 'FC-AIDE'],
                        help='(default=%(default)s)')
    parser.add_argument('--BSN-type', default='normal-BSN', type=str, required=False,
                        choices=['normal-BSN','no-BSN','slightly-BSN','prob-BSN'],
    )
    parser.add_argument('--BSN-param', default=0.1, type=float,
             help="for slightly BSN, it becomes main-pixel multipler, for randomly BSM, it become probability to mask", required=False)
    parser.add_argument('--prob-BSN-test-mode', action='store_true', help='for prob-BSN, test mode on')
    parser.add_argument('--data-type', default=None, type=str, required=False,
                        choices=['Grayscale',
                                 'RawRGB',
                                 'FMD',],
                        help='(default=%(default)s)')
    parser.add_argument('--data-name', default=None, type=str, required=False,
                        choices=['BSD',
                                 'fivek',
                                 'SIDD',
                                 'DND',
                                 'CF_FISH',
                                 'CF_MICE',
                                 'TP_MICE',
                                 'Samsung'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--drop-epoch', default=10, type=int, help='(default=%(default)f)')
    parser.add_argument('--crop-size', default=120, type=int, help='(default=%(default)f)')
    
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.02, type=float, help='(default=%(default)f)')
    parser.add_argument('--test-alpha', default=0, type=float, help='(default=%(default)f)')
    parser.add_argument('--test-beta', default=0, type=float, help='(default=%(default)f)')

    # parser.add_argument('--train-set', type=str, nargs="+", help='To specify train set')
    parser.add_argument('--test-wholedataset', action='store_true', help='To test on wholedataset')
    parser.add_argument('--wholedataset-version', default='v1', type=str, 
                            choices = ["None",'v1', 'v2'],
                            help='Select wholedataset version \
                            v1 : SET01~SET04, v2 : SET05~SET10(default=%(default)f)')

    parser.add_argument('--num-layers', default=8, type=int, help='(default=%(default)f)')
    parser.add_argument('--num-filters', default=64, type=int, help='(default=%(default)f)')
    parser.add_argument('--mul', default=1, type=int, help='(default=%(default)f)')
    
    
    parser.add_argument('--unet-layer', default=3, type=int, help='(default=%(default)f)')
    parser.add_argument('--pge-weight-dir', default=None, type=str, help='(default=%(default)f)')
    
    parser.add_argument('--output-type', default='sigmoid', type=str, help='(default=%(default)f)')
    parser.add_argument('--sigmoid-value', default=0.1, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--use-other-target', action='store_true', help='For samsung SEM image, use other noisy image as target. \
        In PGE-Net evaluation, it denotes specific f number (noisy level), not all F numbers')
    parser.add_argument('--x-f-num', default='F#', type=str, help='For samsung SEM image, set input of f-number 8,16,32,64',
                        choices=['F#', 'F1','F2','F4','F8','F8', 'F01','F02','F04','F08','F16','F32','F64'])
    parser.add_argument('--y-f-num', default='F64', type=str, help='For samsung SEM image, set target of f-number 8,16,32,64',
                        choices=['F1','F2','F4','F8','F01','F02','F04','F08','F16','F32','F64'])
    parser.add_argument('--integrate-all-set', action='store_true', help='For samsung SEM image, no matter what f-number is, integrate all set')
    parser.add_argument('--individual-noisy-input', action='store_true', help='For samsung SEM image, no matter what f-number is, integrate all set')
    parser.add_argument('--dataset-type', default='train', type=str, help='For samsung SEM image, train dataset or test dataset',
                        choices=['train','test','val'])
    parser.add_argument('--set-num', default=-1, type=int, help='For samsung SEM image, need f-number 8,16,32,64',
                        choices=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--test', action='store_true', help='For samsung SEM image, train dataset to be test dataset(small size)')
    parser.add_argument('--log-off',action='store_true', help='logger (neptune) off')
    parser.add_argument('--save-whole-model',action='store_true', help='save whole model')
    parser.add_argument('--speed-test',action='store_true', help='for speed test')
    parser.add_argument('--apply_median_filter',action='store_true', help='apply median_filter instead of FBI-Net')
    parser.add_argument('--apply_median_filter_target',action='store_true', help='apply median_filter to target image')
    parser.add_argument('--train-with-MSEAffine', action='store_true', help='For samsung SEM image, clean image is denoised image with MSE_AFFINE,not F64 image')
    parser.add_argument('--with-originalPGparam', action='store_true', help='For noise estimation, not using PGE-Net, use original PG param(oracle)')
    args=parser.parse_args()
    
    return args




