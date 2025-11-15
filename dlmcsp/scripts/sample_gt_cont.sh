python -m dlmcsp.scripts.sample_gt_cont \
  --ckpt /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.best.pt \
  --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
  --outdir /home/jmmu/dlmcsp/samples \
  --device cuda \
  --formula "GaTe" \
  --spacegroup 194 \
  --wyckoff_letters "4f,4f" \
  --elements "Ga,Te" \
  --t 0.15