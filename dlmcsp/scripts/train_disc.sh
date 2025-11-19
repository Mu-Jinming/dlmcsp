python -m dlmcsp.scripts.train_disc \
  --train_data /home/jmmu/dlmcsp/dlmcsp/data/mp_20/train.clean.ms.jsonl \
  --val_data   /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
  --vocab      /home/jmmu/dlmcsp/configs/vocab.yaml \
  --save       /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.pt \
  --batch 48 --epochs 20 --lr 5e-4 --hidden 512 --layers 12 --heads 8