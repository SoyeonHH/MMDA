cd MISA

python confidNet.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_confidNet True --conf_lr 1e-5 --conf_dropout 0.6 --use_kt False --device cuda:1 --batch_size 32 --n_epoch 10 --n_epoch_conf 10