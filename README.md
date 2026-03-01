#  Emotion Analysis of Stack Overflow Comments
### Using `SamLowe/roberta-base-go_emotions` Transformer Model

> **Course Project** — Semantic & Social Web | Ionian University, Department of Informatics  
> **Author:** Stergios Moutzikos  

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Results & Visualizations](#results--visualizations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project performs a comprehensive **Emotion Analysis** on Stack Overflow user comments using the pre-trained Transformer model [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) from Hugging Face. The model classifies text into **27 fine-grained emotion categories** based on the GoEmotions dataset.

The analysis investigates:
- The **distribution of emotions** across Stack Overflow comments
- **Correlations** between emotions and content popularity (post/comment scores)
- **Geographical differences** in emotional expression
- The relationship between **user reputation** and expressed emotions
- How **upvotes/downvotes** relate to specific emotions

---

## Dataset

Data was sourced from the **Google BigQuery Stack Overflow public dataset** via a Kaggle environment. Three interconnected datasets were sampled and used:

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `comments.csv` | User comments (text, scores, post/user IDs) | ~21,216 | 5 |
| `posts_answers.csv` | Post metadata (scores, comment counts) | ~9,973 | 4 |
| `users.csv` | User metadata (reputation, location, votes) | ~17,384 | 7 |
| `comments_with_emotions.csv` | Comments enriched with emotion predictions | ~20,958 | +1 emotions column |

>  **Note:** `comments_with_emotions.csv` is **not included** in this repository due to its large file size (>31MB). You can regenerate it by running the notebook, or download it separately if provided via an external link.

---





## Methodology

### Pipeline Overview

```
Raw BigQuery Data
       │
       ▼
  Data Loading & EDA
  (missing values, duplicates, distributions)
       │
       ▼
  Text Cleaning
  (lowercase, remove URLs/emojis, filter short comments)
       │
       ▼
  NLP Feature Extraction
  (tokenization, lemmatization, stopword removal, TF-IDF, bigrams)
       │
       ▼
  Emotion Inference
  (SamLowe/roberta-base-go_emotions — 27 emotion classes, threshold=0.3)
       │
       ▼
  Data Merging
  (join comments + users + posts on user_id / post_id)
       │
       ▼
  Analysis & Visualization
  (distributions, heatmaps, box plots, geographical analysis)
```

### Text Preprocessing Steps
1. **Lowercasing** — Normalize case
2. **URL removal** — Strip hyperlinks
3. **Emoji removal** — Remove unicode emoji characters
4. **Deduplication** — Remove duplicate comments
5. **Minimum word filter** — Keep comments with ≥ 3 words
6. **Meaningful text filter** — Require at least one word with 3+ alphanumeric characters

### Emotion Classification
- **Model:** `SamLowe/roberta-base-go_emotions` (RoBERTa fine-tuned on GoEmotions)
- **Top-K:** All 27 classes returned per comment
- **Threshold:** Labels with score > 0.3 are kept
- **Batch size:** 16 (CPU inference)
- **Max tokens:** 512 per comment

### 27 Emotion Categories
`admiration` · `amusement` · `anger` · `annoyance` · `approval` · `caring` · `confusion` · `curiosity` · `desire` · `disappointment` · `disapproval` · `disgust` · `embarrassment` · `excitement` · `fear` · `gratitude` · `grief` · `joy` · `love` · `nervousness` · `optimism` · `pride` · `realization` · `relief` · `remorse` · `sadness` · `surprise` · `neutral`

---

## Key Findings

### 1. Dominance of Neutrality
The most frequent label is **neutral** (15,039 occurrences), consistent with Stack Overflow's technical, information-focused nature.

### 2. Top Non-Neutral Emotions
| Rank | Emotion | Count |
|------|---------|-------|
| 1 | Curiosity | 3,738 |
| 2 | Confusion | 2,848 |
| 3 | Gratitude | 2,523 |
| 4 | Approval | 1,627 |
| 5 | Disapproval | 1,031 |

### 3. Emotion vs. Popularity
- **Higher-scoring comments** tend to carry **positive emotions** (gratitude, admiration, approval)
- **Lower-scoring or downvoted comments** are more associated with **disapproval, disappointment, and annoyance**

### 4. Most Confident Predictions
The model was most confident (>99%) on **gratitude** expressions such as:
> *"Thanks a lot, I will give this a try."*

### 5. Geographical Patterns
- **Germany** and **India** are the top contributing locations
- **Curiosity** and **confusion** are universally dominant across all top-10 locations
- Slight regional variation exists in approval and disapproval rates

---

## Results & Visualizations

The notebook generates the following plots:

| Visualization | Description |
|---------------|-------------|
| Distribution of Post Scores | Histogram of post popularity |
| Distribution of User Reputation | Right-skewed reputation distribution |
| Top 10 User Locations | Bar chart by comment volume |
| Word Count Distribution | Comment length histogram |
| Top 20 Most Frequent Words | Bar chart (stopwords excluded) |
| Top 20 Most Frequent Bigrams | Bar chart of word pairs |
| Frequency of All 27 Emotions | Full emotion bar chart |
| Frequency of Emotions (Excl. Neutral) | Non-neutral emotion bar chart |
| Emotion Distribution by Comment Score | Countplot per score group |
| Emotion Distribution by Post Score | Countplot per post score group |
| Top 10 Locations by Comments | Bar chart |
| Emotions Heatmap by Location | YlGnBu heatmap (with & without neutral) |
| Upvotes by Emotion | Box plot |
| Downvotes by Emotion | Box plot |

---

## Limitations

- **Model dependency:** The GoEmotions model may struggle with technical jargon, sarcasm, or irony common in developer communities
- **Data bias:** Random 0.025% sampling may not perfectly represent the full Stack Overflow community
- **No temporal analysis:** Timestamps were not included, so time-based trends cannot be assessed
- **Age data missing:** 100% of the `age` column in users was null

---

## Future Work

- **Temporal trends:** Incorporate timestamps to track emotional trends over time
- **Aspect-Based Sentiment Analysis (ABSA):** Identify emotions directed at specific topics (e.g., a language or framework)
- **Sarcasm/irony detection:** Improve accuracy for indirect expressions
- **User emotion profiles:** Build per-user emotional fingerprints
- **Cross-community comparison:** Compare Stack Overflow vs. Reddit vs. GitHub Issues
- **Multilingual support:** Extend to non-English comments

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `transformers` (Hugging Face) | GoEmotions model |
| `torch` | Deep learning backend |
| `nltk` | Tokenization, lemmatization, stopwords |
| `scikit-learn` | TF-IDF vectorization |
| `gensim` | Word2Vec / bigram support |
| `tqdm` | Progress tracking |
| `emoji` | Emoji detection and removal |
| `textblob` | Supplementary NLP |
| `nrclex` | NRC emotion lexicon support |

---

## References

Key references (full bibliography in `Report_with_Code_Appendix.pdf`):

- Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. EMNLP 2020.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL 2019.
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv.
- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
- Pang & Lee (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in IR.

---

## License

This project was created for academic purposes as part of the **Semantic & Social Web** course at Ionian University. Please cite appropriately if reusing.

---

*For questions, feel free to open an issue or reach out via the repository.*
