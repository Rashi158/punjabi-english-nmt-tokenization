# Morphology-Aware Tokenization for Low-Resource Punjabi–English Machine Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rashi158/datasciproj/blob/main/01_setupanddata.ipynb)

Does splitting Punjabi words at real morpheme boundaries help a translation model when training data is scarce? This project builds a graph-based pretokenizer for Punjabi and compares it with standard subword tokenizers by training 16 Transformer models on 10K, 25K and 50K sentence pairs.

## Approach

1. **Data.** 300K Punjabi–English pairs from [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) are normalized, filtered and deduplicated down to 244,910 clean pairs. Nested 10K/25K/50K training splits are drawn from them, with zero overlap with the FLORES-200 test set.
2. **Graph-based pretokenizer.** Every possible stem + suffix split of each Punjabi word becomes an edge in a bipartite graph. A seeded HITS algorithm scores stems and suffixes against each other, so genuine suffixes rise to the top (for example, -ਦਾ moves from rank 25 to rank 9). Words are then cut at the best-scoring boundary before BPE runs.
3. **Baselines.** BPE, Unigram LM and Morfessor + BPE, all with the same vocabulary size. Only the Punjabi side changes; the English tokenizer is fixed.
4. **Models.** One Transformer NMT model (9.6M parameters) per tokenizer and data size, written in PyTorch and trained on a Colab T4 GPU.
5. **Evaluation.** BLEU and chrF++ with sacreBLEU on FLORES-200, plus paired bootstrap resampling (2,000 samples) to test each method against BPE.

## Results

Test scores on FLORES-200 (Punjabi → English).

| Method | BLEU 10K | BLEU 25K | BLEU 50K | chrF++ 10K | chrF++ 25K | chrF++ 50K |
| --- | --- | --- | --- | --- | --- | --- |
| BPE | 1.16 | 2.27 | 3.08 | 18.19 | 20.05 | 21.48 |
| Unigram LM | 1.36 | 2.41 | 3.10 | 18.36 | 20.22 | 21.33 |
| Morfessor + BPE | 1.09 | 1.93 | 2.86 | 17.75 | 19.47 | 21.37 |
| Graph, no seeds | 1.17 | – | – | 18.16 | – | – |
| Graph, rare words (ours) | 1.38 | 2.32 | 3.05 | 18.23 | 20.04 | 21.75 |
| Graph, all words (ours) | 1.41 | 2.34 | 3.02 | 17.88 | 20.65 | 21.63 |

What the numbers say:

- **Data size matters more than the tokenizer.** Going from 10K to 50K pairs adds about 3.3 chrF++ for every method, while the gap between methods at any one size stays under 1.2 chrF++.
- **The graph method gives one statistically significant gain:** +0.66 chrF++ over BPE at 25K pairs (95% interval +0.30 to +1.02).
- **Morfessor + BPE is significantly worse than BPE** at 10K and 25K pairs, partly because it produces much longer sequences (about 46–48 tokens per sentence against 38).
- Absolute scores are low because the models are small and trained from scratch on very little data. The comparison between tokenizers, not the absolute score, is the point of the study.

## Run it

Open `01_setupanddata.ipynb` in Google Colab with a GPU runtime and run the cells in order. The notebook installs its own dependencies, downloads Samanantar and FLORES-200, and saves checkpoints and results to Google Drive so training can resume after a disconnect. The 50K runs take roughly 1.5 to 2 hours on a T4.

## Tech stack

Python, PyTorch, SentencePiece, Morfessor, NetworkX, sacreBLEU, pandas, Hugging Face Datasets, matplotlib
