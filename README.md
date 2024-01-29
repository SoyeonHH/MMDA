# Multi-modal Multi-label Dynamic Adaptation

## 1. MOSI variable for Emotion Classification

* Dataset: CMU-MOSEI
* Source: textual feature, visual feature, acoustic feature
* Target: sentiment label (-3, 3), emotion label (6-class)
* Emotion 6-class: ​​​​{​​​​​happiness, sadness, anger, fear, disgust, surprise}

## Before Running

1. Clone the repo.

```bash
git clone git@github.com:SoyeonHH/MMDA.git
```

2. Set your CUDA device in [here](https://github.com/SoyeonHH/MMDA/blob/43b677d810ee3a2285799dd5af185df62182fa85/src/config.py#L11)

```python
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
```

3. Modify the downloaded [GloVe](https://nlp.stanford.edu/projects/glove/) file path in [here](https://github.com/SoyeonHH/MMDA/blob/43b677d810ee3a2285799dd5af185df62182fa85/src/config.py#L15)

```python
word_emb_path = '/data1/multimodal/glove.840B.300d.txt'
```

4. Modify [CMU-Multimodal SDK](https://github.com/ecfm/CMU-MultimodalSDK) file path in [here](https://github.com/SoyeonHH/MMDA/blob/43b677d810ee3a2285799dd5af185df62182fa85/src/config.py#L21)

```python
sdk_dir = Path('/data1/multimodal/CMU-MultimodalSDK')
```

5. Modify MOSI and MOSEI file path in [here](https://github.com/SoyeonHH/MMDA/blob/43b677d810ee3a2285799dd5af185df62182fa85/src/config.py#L22) with downloaded dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK).

```python
data_dir = Path('/data1/multimodal')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath('MOSEI')}
```


## Run

```bash
bash train.sh
```

If you want to use additional network, called 'ConfidNet', in previous architecture, run this code:

```bash
bash train_confid.sh
```

## Acknowledgement

The Dataset source is from [CMU-Multimodal SDK](https://github.com/ecfm/CMU-MultimodalSDK), [kniter1/TAILOR](https://github.com/kniter1/TAILOR), and [declare-lab/Multimodal-Infomax](https://github.com/declare-lab/Multimodal-Infomax).
