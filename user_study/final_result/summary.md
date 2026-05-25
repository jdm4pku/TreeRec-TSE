# User Study - Final Results

## 1. Overall Scores (All 72 Queries)

| Method | Avg Relevance | Std | Avg Usefulness | Std |
|--------|---------------|-----|----------------|-----|
| BM25 | 3.22 | 1.17 | 3.14 | 1.13 |
| GPT4 | 4.17 | 1.01 | 4.11 | 1.03 |
| TreeRec | 4.82 | 0.51 | 4.78 | 0.61 |

## 2. Scores by Ecosystem

### JavaScript (36 Queries)

| Method | Avg Relevance | Std | Avg Usefulness | Std |
|--------|---------------|-----|----------------|-----|
| BM25 | 3.06 | 1.25 | 2.92 | 1.19 |
| GPT4 | 4.19 | 1.20 | 4.08 | 1.21 |
| TreeRec | 4.69 | 0.66 | 4.61 | 0.79 |

### HuggingFace (36 Queries)

| Method | Avg Relevance | Std | Avg Usefulness | Std |
|--------|---------------|-----|----------------|-----|
| BM25 | 3.39 | 1.06 | 3.36 | 1.03 |
| GPT4 | 4.14 | 0.79 | 4.14 | 0.82 |
| TreeRec | 4.94 | 0.23 | 4.94 | 0.23 |

## 3. Wilcoxon Signed-Rank Tests

| Comparison | Dimension | Statistic | p-value | Significance |
|------------|-----------|-----------|---------|--------------|
| TreeRec vs GPT4 | relevance | 40.5 | 0.0000 | *** |
| TreeRec vs GPT4 | usefulness | 75.0 | 0.0000 | *** |
| TreeRec vs BM25 | relevance | 0.0 | 0.0000 | *** |
| TreeRec vs BM25 | usefulness | 0.0 | 0.0000 | *** |
| GPT4 vs BM25 | relevance | 111.5 | 0.0000 | *** |
| GPT4 vs BM25 | usefulness | 151.0 | 0.0000 | *** |

## 4. Overall Ranking (72 Queries)

| Method | Best | Neutral | Worst |
|--------|------|---------|-------|
| BM25 | 3 | 9 | 60 |
| GPT4 | 17 | 43 | 12 |
| TreeRec | 52 | 20 | 0 |

### Ranking Percentage

| Method | Best% | Neutral% | Worst% |
|--------|-------|----------|--------|
| BM25 | 4.2% | 12.5% | 83.3% |
| GPT4 | 23.6% | 59.7% | 16.7% |
| TreeRec | 72.2% | 27.8% | 0.0% |
