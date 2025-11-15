python -m dlmcsp.scripts.sample_cont \
  --data /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
  --ckpt  /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.best.pt \
  --device cuda \
  --out_ms /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.test_samples.jsonl