python -m dlmcsp.scripts.compute_metrics \
  --pred_ms /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.test_samples.jsonl \
  --gt_ms /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab configs/vocab.yaml \
  --only_geom_ok \
  --require_same_formula