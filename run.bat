call activate
@REM for %%a in (3, 7, 11, 13) do (
@REM     python main_allsite.py --path all_sites --resultspath CNN1_ker1_%%a --epochs 500 --lr 0.001 --batch 1024 --datapre dataset_all.mat --ker1 %%a --ker2 11 --step 1 --nums 256
@REM )
@REM for %%b in (3, 7, 9, 13) do (
@REM     python main_allsite.py --path all_sites --resultspath CNN1_ker2_%%b --epochs 500 --lr 0.001 --batch 1024 --datapre dataset_all.mat --ker1 5 --ker2 %%b --step 1 --nums 256
@REM )
@REM for %%c in (256, 128, 1024) do (
@REM     python main_allsite.py --path all_sites --resultspath CNN1_nums_%%c --epochs 500 --lr 0.001 --batch 1024 --datapre dataset_all.mat --ker1 5 --ker2 11 --step 1 --nums %%c
@REM )
@REM for %%d in (0.01, 0.0005, 0.0001) do (
@REM     python main_allsite.py --path all_sites --resultspath CNN1_lr_%%d --epochs 500 --lr %%d --batch 1024 --datapre dataset_all.mat --ker1 5 --ker2 11 --step 1 --nums 512
@REM )
@REM for %%e in (256, 512, 2048, 4069) do (
@REM     python main_allsite.py --path all_sites --resultspath CNN1_batch_%%e --epochs 500 --lr 0.001 --batch %%e --datapre dataset_all.mat --ker1 5 --ker2 11 --step 1 --nums 512
@REM )
python main_allsite.py --path all_sites --resultspath CNN_HPO_MS --epochs 1000 --lr 0.0014 --batch 2593 --datapre dataset_all2.mat --ker1 7 --ker2 41 --step 1 --nums 887
@REM python main_allsite.py --path all_sites --resultspath CNN_extra_all2 --epochs 1000 --lr 0.001 --batch 1024 --datapre dataset.mat
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_FKSH12_strong1 --epochs 500 --lr 0.001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_FKSH12
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_IBRH13_strong1 --epochs 500 --lr 0.001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_IBRH13
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_IBRH10_strong1 --epochs 500 --lr 0.001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_IBRH10
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_IBRH14_strong1 --epochs 500 --lr 0.000001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_IBRH14
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_IWTH14_strong1 --epochs 500 --lr 0.000001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_IWTH14
@REM python main_retrain.py --path all_sites --resultspath CNN_extra_all1_MYGH10_strong1 --epochs 500 --lr 0.000001 --batch 1 --datapre dataset.mat --pretrain CNN_extra_all1_MYGH10
@REM python main_allsite.py --path all_sites --resultspath CNN_IBRH11 --epochs 1000 --lr 0.0001 --batch 16 --datapre dataset.mat
pause