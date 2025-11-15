python -m dlmcsp.scripts.train \
  --train_data /home/jmmu/dlmcsp/dlmcsp/data/mp_20/train.clean.ms.jsonl \
  --val_data   /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab      /home/jmmu/dlmcsp/configs/vocab.yaml \
  --save       /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.pt \
  --epochs 30 \
  --batch 64 \
  --lr 5e-5