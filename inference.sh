## MISA: Modality-Invariant and -Specific Representation Learning for Multimodal Emotion Recognition
cd MISA

python inference.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt False --device cuda --batch_size 32 --checkpoint '/home/soyeon/workspace/MMDA/checkpoints/model_2023-03-20_11:20:30.std'

python inference.py --data mosei --eval_mode micro --learning_rate 1e-5 --dropout 0.6 --use_kt True --kt_model Static --kt_weight 100.0 --device cuda:1 --batch_size 32 --checkpoint '/home/soyeon/workspace/MMDA/checkpoints/model_2023-03-20_11:18:48.std'
