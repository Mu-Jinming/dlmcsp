python -m dlmcsp.scripts.sample_disc_remask \
  --data /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
  --ckpt /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.best.pt \
  --out  /home/jmmu/dlmcsp/dlmcsp/outputs/test.disc.pred.jsonl \
  --steps 16 --r_init 1.0 --r_final 0.1 --gamma 1.0