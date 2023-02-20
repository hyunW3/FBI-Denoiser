import os
import subprocess
alpha_beta = [[0.05,0.02],[0.01,0.02],[0.01,0.0002]]

for alpha, beta in alpha_beta:
    #print(alpha,beta)
    main_cmd = 'knockknock telegram \
        --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
        --chat-id 1597147353 \
        ./train_fbi_net_synthetic_noise.sh '
    argument = f'{alpha} {beta} '
    log_file = f'2>&1 | tee out_alpha{alpha}_beta_{beta}.txt'
    cmd = main_cmd + argument + log_file
    cmd = list(filter(lambda x : x != '',cmd.split(" ")))
    print(f"===== run fivek alpha : {alpha} beta : {beta} =====")
    #print(cmd)
    # result = subprocess.run(cmd)
