# RecoSphere Evaluation Results

## Rating prediction (CF)

| Domain | RMSE | MAE |
|---|---|---|
| Movie | 1.1255 | 0.9439 |
| Book | 0.9916 | 0.7754 |

## Movie — ranking quality (@10)

| Method | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| CF only | 0.0253 | 0.0223 | 0.0302 |
| Content only | 0.0180 | 0.0221 | 0.0248 |
| Hybrid | 0.0297 | 0.0309 | 0.0362 |

(evaluated on 300 users)

## Book — ranking quality (@10)

| Method | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| CF only | 0.0550 | 0.0383 | 0.0634 |
| Content only | 0.0347 | 0.0231 | 0.0402 |
| Hybrid | 0.0433 | 0.0291 | 0.0504 |

(evaluated on 300 users)

## Cross-domain cold-start (movie -> book) vs popularity baseline

| Method | Precision@10 |
|---|---|
| Cross-domain (shared embedding) | 0.0050 |
| Popularity baseline | 0.2127 |

(evaluated on 300 users with ratings in both domains)
