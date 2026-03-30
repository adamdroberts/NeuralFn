import json
from neuralfn.torch_templates import build_gpt_template_payload

def test():
    print("Testing MoE Payload Generation...")
    payload = build_gpt_template_payload("gpt", {"preset": "moe"})
    vl = payload.get("variant_library", {})
    families = list(vl.keys())
    print(f"Variant Families Generated: {families}")
    if "moe" not in families:
        print("ERROR: 'moe' not found in variant library!")
        return
    print("SUCCESS")
test()
