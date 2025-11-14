python -m dlmcsp.scripts.train_llada \
  --data /home/jmmu/dlmcsp/dlmcsp/data/mp_20/train.ms.jsonl \
  --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
  --save /home/jmmu/dlmcsp/ckpts/llada_gt_s512l8h8.pt \
  --batch 24 --hidden 512 --layers 8 --heads 8 --steps 50000