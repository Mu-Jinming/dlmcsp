# check_reconstruction.py
import json
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from dlmcsp.representation.ms_to_cif import ms_to_structure
from dlmcsp.tokenization.vocab_utils import load_vocab_yaml, get_lattice_conf, inv_bin_lattice_scalar, get_param_conf, inv_quantize_param
import multiprocessing
import os

def _load_jsonl(path: str):
    # (从 sample.py 复制这个函数)
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            recs.append(json.loads(ln))
    return recs

def reconstruct_from_discrete_only(ms: dict, vocab_path: str) -> dict:
    """
    创建一个新的 ms 字典，强制从离散的 bin/token 反量化，忽略原始的 'value' 字段。
    这模拟了模型生成离散 token 后的真实情况。
    """
    new_ms = json.loads(json.dumps(ms)) # Deep copy
    vocab = load_vocab_yaml(vocab_path)
    latt_conf = get_lattice_conf(vocab)
    param_conf = get_param_conf(vocab)
    # 1. 强制反量化晶格参数
    for name, conf in latt_conf.items():
        bin_idx = new_ms["latt"][name]["bin"]
        dr = new_ms["latt"][name].get("dr", 0)
        new_ms["latt"][name]["value"] = inv_bin_lattice_scalar(bin_idx, dr, conf)
    # 2. 强制反量化 Wyckoff 参数
    for site in new_ms.get("sites", []):
        params = site.get("params", {})
        if params == "-" or params is None: continue
        for k in ("u", "v", "w"):
            if k in params:
                # 假设 params[k] 是一个包含 mode, base/idx 等信息的字典
                params[k]["value"] = inv_quantize_param(params[k], conf=param_conf)
               
    return new_ms

def worker(args):
    ms_gt, vocab_path = args
    try:
        # 结构A: 从带有精确 'value' 的原始 GT 构建 (作为基准真值)
        struct_gt = ms_to_structure(ms_gt, vocab_path)
       
        # 结构B: 从只使用离散 token 反量化的 ms 构建
        ms_discrete = reconstruct_from_discrete_only(ms_gt, vocab_path)
        struct_reconstructed = ms_to_structure(ms_discrete, vocab_path)
       
        matcher = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)
        return matcher.fit(struct_gt, struct_reconstructed)
    except Exception as e:
        print(f"Failed on material_id {ms_gt.get('material_id')}: {e}")
        return False

def main():
    gt_path = "/home/jmmu/dlmcsp/dlmcsp/data/mp_20/test.clean.ms.jsonl" # 你的测试集
    vocab_path = "/home/jmmu/dlmcsp/configs/vocab.yaml"
   
    gt_recs = _load_jsonl(gt_path)
    
    num_processes = os.cpu_count()-4  # 使用所有可用CPU核心
    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(ms_gt, vocab_path) for ms_gt in gt_recs]
        results = list(tqdm(pool.imap(worker, args), total=len(gt_recs), desc="Checking Reconstruction Limit"))
    
    n_total = len(gt_recs)
    n_match = sum(1 for r in results if r)
    
    match_rate = n_match / n_total if n_total > 0 else 0
    print("="*40)
    print("RECONSTRUCTION TEST SUMMARY")
    print(f"Total samples tested: {n_total}")
    print(f"Successfully matched: {n_match}")
    print(f"Theoretical Max Matching Rate: {match_rate:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()