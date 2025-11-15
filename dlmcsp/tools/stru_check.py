from dlmcsp.representation.ms_to_cif import ms_to_structure
from dlmcsp.eval.validators import quick_validate_structure
import json

ms = json.load(open("/home/jmmu/dlmcsp/samples/sample_000.ms.json"))
struct = ms_to_structure(ms, "/home/jmmu/dlmcsp/configs/vocab.yaml")
ok, why = quick_validate_structure(struct)
print(ok, why)
