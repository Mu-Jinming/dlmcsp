python -m dlmcsp.scripts.compute_metrics \
  --pred_ms /home/jmmu/dlmcsp/dlmcsp/outputs/test.disc.pred.jsonl \
  --gt_ms /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab configs/vocab.yaml \
  --require_same_formula