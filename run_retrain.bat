call activate
@REM for %%s in (WLA) do (
@REM     python main_retrain.py --path all_sites --resultspath CNN_HPO_%%s --epochs 500 --lr 0.00001 --batch 16 --datapre dataset.mat --pretrain CNN_HPO --ker1 7 --ker2 41 --step 1 --nums 887
@REM )
for %%s in (IBRH14, FKSH12, IBRH13, IBRH10, IWTH14, MYGH10) do (
    python main_retrain.py --path all_sites --resultspath CNN_HPO_%%s_train20 --epochs 500 --lr 0.00001 --batch 10 --datapre dataset.mat --pretrain CNN_HPO --ker1 7 --ker2 41 --step 1 --nums 887
)
@REM for %%s in (IBRH14, FKSH12, IBRH13, IBRH10, IWTH14, MYGH10) do (
@REM     python main_retrain.py --path all_sites --resultspath CNN_HPO_%%s_scratch --epochs 500 --lr 0.001 --batch 10 --datapre dataset.mat  --ker1 7 --ker2 41 --step 1 --nums 887
@REM )
@REM for %%s in (IBRH14, FKSH12, IBRH13, IBRH10, IWTH14, MYGH10) do (
@REM     python main_retrain.py --path all_sites --resultspath CNN_HPO_%%s_strong --epochs 500 --lr 0.000001 --batch 1 --datapre dataset.mat --pretrain CNN_HPO_%%s --ker1 7 --ker2 41 --step 1 --nums 887
@REM )
pause