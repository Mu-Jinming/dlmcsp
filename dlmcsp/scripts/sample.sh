python -m dlmcsp.scripts.sample \
    --ckpt /home/jmmu/dlmcsp/dlmcsp/ckpts/llada_cont.best.pt \
    --vocab /home/jmmu/dlmcsp/configs/vocab.yaml \
    --device cuda \
    --cond_ms /home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl \
    --outdir /home/jmmu/dlmcsp/dlmcsp/out \
    --out_jsonl /home/jmmu/dlmcsp/dlmcsp/out/llada_cont.test_samples.jsonl \
    --t 0.15 \
    --limit 200

