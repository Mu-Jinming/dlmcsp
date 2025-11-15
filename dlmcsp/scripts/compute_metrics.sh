python -m dlmcsp.scripts.compute_metrics \
  --pred_ms /home/jmmu/dlmcsp/dlmcsp/out/llada_cont.test_samples.jsonl \
  --gt_ms /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab configs/vocab.yaml \
  --require_same_formula