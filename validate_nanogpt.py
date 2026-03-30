import json
from neuralfn.torch_templates import build_gpt_template_payload

print('=== NanoGPT Validation ===')
payload = build_gpt_template_payload('gpt', {'preset': 'nanogpt'})
vl = payload['variant_library']

# Check SDPA config for dropout_p
attn_graph = vl['attention']['default']
sdpa_node = attn_graph['nodes']['sdpa']
sdpa_cfg = sdpa_node['neuron_def']['module_config']
print(f'1. SDPA is_causal: {sdpa_cfg.get("is_causal")}')
print(f'   SDPA dropout_p: {sdpa_cfg.get("dropout_p")}')

# Check position embedding gets wired to tokens_in (not token_embed)
model_sg = payload['node_def']['subgraph']
pos_embed_edges = [e for e in model_sg['edges'].values() if e['dst_node'] == 'pos_embed']
for e in pos_embed_edges:
    print(f'2. pos_embed input from: {e["src_node"]} port {e["src_port"]}')

# Check tied_lm_head wiring
tie_edges = [e for e in model_sg['edges'].values() if e['dst_node'] == 'tied_lm_head']
for e in tie_edges:
    print(f'3. tied_lm_head input from: {e["src_node"]} port {e["src_port"]} -> port {e["dst_port"]}')

# Check softcap presence
has_softcap = 'softcap' in model_sg['nodes']
print(f'4. Softcap node present: {has_softcap}')

# Check CE edge source
ce_edges = [e for e in model_sg['edges'].values() if e['dst_node'] == 'ce']
for e in ce_edges:
    print(f'   CE input from: {e["src_node"]} port {e["src_port"]} -> port {e["dst_port"]}')

print()
print('=== Summary ===')
print(f'  SDPA is_causal=True: {sdpa_cfg.get("is_causal") == True}')
print(f'  SDPA dropout_p > 0:  {sdpa_cfg.get("dropout_p", 0) > 0}')
print(f'  pos_embed from tokens (not embed): {pos_embed_edges[0]["src_node"] == "tokens_in"}')
tie_weight_edge = [e for e in tie_edges if e['dst_port'] == 1]
print(f'  tied_lm_head gets embedding.weight: {len(tie_weight_edge) > 0 and tie_weight_edge[0]["src_node"] == "token_embed" and tie_weight_edge[0]["src_port"] == 1}')
print(f'  Softcap disabled (no node): {not has_softcap}')
