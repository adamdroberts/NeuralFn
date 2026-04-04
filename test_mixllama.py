import json
from neuralfn.torch_templates import build_gpt_template_payload

def test():
    print("Testing MoE Payload Generation...")
    payload = build_gpt_template_payload("gpt", {"preset": "mixllama"})
    vl = payload.get("variant_library", {})
    families = list(vl.keys())
    print(f"Variant Families Generated: {families}")
    if "mixllama" not in families:
        print("ERROR: 'mixllama' not found in variant library!")
        return
    print("SUCCESS")
test()
