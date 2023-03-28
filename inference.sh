## MISA: Modality-Invariant and -Specific Representation Learning for Multimodal Emotion Recognition
cd MISA

# python inference.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt False --device cuda --batch_size 32 --checkpoint '/home/soyeon/workspace/MMDA/checkpoints/model_2023-03-20_11:20:30.std'

python inference.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_confidNet True --conf_lr 1e-5 --conf_dropout 0.6 --use_kt True --kt_model Dynamic-tcp --kt_weight 10000.0 --dynamic_method ratio --device cuda --batch_size 32
