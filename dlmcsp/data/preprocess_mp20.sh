python -m dlmcsp.data.preprocess_mp20 \
  --csv /home/jmmu/dlmcsp/dlmcsp/data/mp/test.csv \
  --out /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.ms.jsonl \
  --vocab configs/vocab.yaml \
  --setting hex \
  --num_workers 12 \