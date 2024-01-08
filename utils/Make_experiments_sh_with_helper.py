import os

def explore_the_experimetns_and_generate_nohup_lines(path='/root/autodl-tmp/PuzzleTuning_Comparison'):
    data_augmentation_dic = {'ROSE': '0', 'pRCC': '0', 'CAM16': '3', 'WBC': '3'}
    for exp_root in os.listdir(path):
        out_sh_name = exp_root + '.sh'
        lr_mystr = exp_root.split('_')[0]
        lrf_mystr = exp_root.split('_')[1].split('lf')[-1]
        dataset_name = exp_root.split('_')[-1]
        data_augmentation_mode = data_augmentation_dic[dataset_name]
        print('nohup python Experiment_script_helper.py --lr_mystr ' + lr_mystr + ' --lrf_mystr ' + lrf_mystr
              + ' --data_augmentation_mode ' + data_augmentation_mode + ' --dataset_name ' + dataset_name + ' > '
              + out_sh_name + ' 2>&1 &')
        
        
explore_the_experimetns_and_generate_nohup_lines('/Users/zhangtianyi/Downloads/PuzzleTuning_Comparison')
'''
nohup python Experiment_script_helper.py --lr_mystr 408 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name ROSE > 408_lf25_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 607 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name WBC > 607_lf05_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 40 --data_augmentation_mode 0 --dataset_name ROSE > 503_lf40_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 605 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name ROSE > 605_lf50_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 606_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 503_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name CAM16 > 503_lf40_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name pRCC > 504_lf25_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 509_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 507_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 607 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 607_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 401 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name pRCC > 401_lf35_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 402 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 402_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 305_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name pRCC > 509_lf25_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 502 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name ROSE > 502_lf50_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 504_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 507_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 30 --data_augmentation_mode 0 --dataset_name ROSE > 504_lf30_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name pRCC > 304_lf35_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 307 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name pRCC > 307_lf20_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 408 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name WBC > 408_lf40_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 605 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name WBC > 605_lf50_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 503_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 303 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name ROSE > 303_lf10_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 302_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name pRCC > 506_lf20_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name pRCC > 608_lf25_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 306_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 501 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 501_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name ROSE > 506_lf35_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name CAM16 > 506_lf40_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 503_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 404 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name WBC > 404_lf25_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name WBC > 609_lf35_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name WBC > 609_lf20_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 404 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name WBC > 404_lf30_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name WBC > 606_lf15_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 607 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name WBC > 607_lf15_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 30 --data_augmentation_mode 0 --dataset_name pRCC > 507_lf30_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 303 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 303_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name CAM16 > 508_lf10_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 409 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name WBC > 409_lf25_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 501 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 501_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 306_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 305_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 604 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name WBC > 604_lf35_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 407 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 407_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 505_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 409 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name ROSE > 409_lf25_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 407 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name WBC > 407_lf05_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 406 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name WBC > 406_lf05_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 40 --data_augmentation_mode 0 --dataset_name pRCC > 507_lf40_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 307 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 307_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 301 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name CAM16 > 301_lf25_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 603 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 603_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name ROSE > 503_lf50_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name ROSE > 508_lf10_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 508_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 304_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name CAM16 > 304_lf40_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 606_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name WBC > 608_lf50_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name WBC > 609_lf50_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name CAM16 > 509_lf25_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 607 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name CAM16 > 607_lf40_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 503_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name WBC > 505_lf15_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name CAM16 > 508_lf35_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 608_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name ROSE > 506_lf25_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 401 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 401_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 501 --lrf_mystr 40 --data_augmentation_mode 0 --dataset_name pRCC > 501_lf40_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name WBC > 608_lf40_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 506_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name pRCC > 503_lf25_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name ROSE > 609_lf10_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 509_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name CAM16 > 305_lf15_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name ROSE > 505_lf50_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name pRCC > 608_lf35_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 402 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name CAM16 > 402_lf10_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 408 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 408_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name CAM16 > 306_lf15_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 507_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 508_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 503_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name ROSE > 302_lf25_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 502 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 502_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name pRCC > 608_lf20_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 40 --data_augmentation_mode 0 --dataset_name pRCC > 509_lf40_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name WBC > 504_lf25_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 301 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name ROSE > 301_lf50_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 302_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 508 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 508_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 506_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 301 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name WBC > 301_lf30_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 507_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 307 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name pRCC > 307_lf05_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 503_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 605 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 605_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 503_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 301 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name WBC > 301_lf35_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 606_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name CAM16 > 304_lf10_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 401 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name CAM16 > 401_lf30_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 304_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name pRCC > 504_lf20_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 504_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 506_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 25 --data_augmentation_mode 0 --dataset_name ROSE > 505_lf25_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 501 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name ROSE > 501_lf20_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 301 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name WBC > 301_lf20_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 502 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name WBC > 502_lf50_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 305_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 401 --lrf_mystr 30 --data_augmentation_mode 0 --dataset_name pRCC > 401_lf30_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 409 --lrf_mystr 30 --data_augmentation_mode 0 --dataset_name ROSE > 409_lf30_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 303 --lrf_mystr 50 --data_augmentation_mode 3 --dataset_name CAM16 > 303_lf50_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 505_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 607 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name ROSE > 607_lf10_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 309 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 309_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 302_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name CAM16 > 306_lf35_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 304_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 605 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 605_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name WBC > 606_lf30_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 502 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name WBC > 502_lf20_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 506 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name pRCC > 506_lf15_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 50 --data_augmentation_mode 0 --dataset_name pRCC > 509_lf50_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name WBC > 609_lf05_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 303 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 303_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 609_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name CAM16 > 304_lf15_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 15 --data_augmentation_mode 0 --dataset_name ROSE > 503_lf15_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 507 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 507_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name WBC > 606_lf25_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 402 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name CAM16 > 402_lf20_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name ROSE > 503_lf35_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 303 --lrf_mystr 05 --data_augmentation_mode 0 --dataset_name ROSE > 303_lf05_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 608 --lrf_mystr 30 --data_augmentation_mode 0 --dataset_name pRCC > 608_lf30_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name ROSE > 302_lf35_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name CAM16 > 306_lf30_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 505_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 40 --data_augmentation_mode 0 --dataset_name pRCC > 302_lf40_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name WBC > 302_lf05_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 409 --lrf_mystr 20 --data_augmentation_mode 0 --dataset_name ROSE > 409_lf20_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 302 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name WBC > 302_lf10_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 306 --lrf_mystr 10 --data_augmentation_mode 3 --dataset_name CAM16 > 306_lf10_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 504_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 304 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name ROSE > 304_lf35_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name WBC > 505_lf40_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 504 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name WBC > 504_lf40_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 503 --lrf_mystr 25 --data_augmentation_mode 3 --dataset_name WBC > 503_lf25_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name ROSE > 606_lf10_ROSE.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 40 --data_augmentation_mode 3 --dataset_name CAM16 > 305_lf40_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 305 --lrf_mystr 05 --data_augmentation_mode 3 --dataset_name CAM16 > 305_lf05_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 609 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name WBC > 609_lf15_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 606 --lrf_mystr 20 --data_augmentation_mode 3 --dataset_name WBC > 606_lf20_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 502 --lrf_mystr 30 --data_augmentation_mode 3 --dataset_name WBC > 502_lf30_WBC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 509 --lrf_mystr 10 --data_augmentation_mode 0 --dataset_name pRCC > 509_lf10_pRCC.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 404 --lrf_mystr 35 --data_augmentation_mode 3 --dataset_name CAM16 > 404_lf35_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 401 --lrf_mystr 15 --data_augmentation_mode 3 --dataset_name CAM16 > 401_lf15_CAM16.sh 2>&1 &
nohup python Experiment_script_helper.py --lr_mystr 505 --lrf_mystr 35 --data_augmentation_mode 0 --dataset_name ROSE > 505_lf35_ROSE.sh 2>&1 &
'''
