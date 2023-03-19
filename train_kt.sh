python src/train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt False --device cuda:1 --batch_size 32 --n_epoch 40

python src/train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt True --kt_model Static --kt_weight 100.0 --device cuda:1 --batch_size 32 --n_epoch 40

python src/train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --conf_lr 1e-5 --conf_dropout 0.6 --use_kt True --kt_model Dynamic-tcp --kt_weight 100.0 --device cuda:1 --batch_size 32 --n_epoch 10 --n_epoch_conf 10 --n_epoch_dkt 30
