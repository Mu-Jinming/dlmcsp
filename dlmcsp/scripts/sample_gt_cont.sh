python -m dlmcsp.scripts.sample_gt_cont \
  --ckpt /home/jmmu/dlmcsp/ckpts/llada_cont_s512l8h8.pt \
  --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
  --outdir /home/jmmu/dlmcsp/samples \
  --device cuda \
  --formula "GaTe" \
  --spacegroup 194 \
  --wyckoff_letters "4f,4f" \
  --elements "Ga,Te" \
  --num 2 --t 0.15