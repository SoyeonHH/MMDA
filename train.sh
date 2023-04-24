## MISA: Modality-Invariant and -Specific Representation Learning for Multimodal Emotion Recognition
cd MISA

# baseline
# python train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt False --use_confidNet False --device cuda:1 --batch_size 32 --n_epoch 40

# Static KT
# python train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_confidNet False --use_kt True --kt_model Static --kt_weight 10000.0 --device cuda:1 --batch_size 32 --n_epoch 40

# DynamicKT-CE
# python train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_confidNet False --use_kt True --kt_model Dynamic-ce --kt_weight 10000.0 --dynamic_method ratio --device cuda:1 --batch_size 32 --n_epoch 40

# DynamicKT-TCP
python train.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_confidNet True --conf_lr 1e-5 --conf_dropout 0.6 --use_kt True --kt_model Dynamic-tcp --kt_weight 100000.0 --dynamic_method ratio --device cuda:1 --batch_size 32 --n_epoch 10 --n_epoch_conf 10 --n_epoch_dkt 30

