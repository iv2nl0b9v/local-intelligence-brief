"""Precision- and stage-specific arithmetic model for speculative decode.
No precision is inferred from weight bits. FMA/MAC is two operations; integer
paths are GOP/TOPS, not floating FLOPs. All ceilings are dense, never 2:4 sparse.
"""
from pathlib import Path
from math import prod
from statistics import median
import json
ROOT=Path(__file__).resolve().parents[1]
INPUTS=json.loads((ROOT/'data/compute-inputs.json').read_text(encoding='utf-8'))

def config(name):
    d=json.loads((ROOT/'data/configs'/name).read_text())
    return d.get('text_config',d)

MODELS={
 'qwen36':dict(active_b=3,layers=40,h=2048,heads=16,attn_layers=10,head_dim=256,cap=None,vocab=248320,routed_b=0),
 'qwen27':dict(active_b=26,layers=64,h=5120,heads=24,attn_layers=16,head_dim=256,cap=None,vocab=248320,routed_b=0),
 'flashnext':dict(active_b=6,layers=48,h=2560,heads=24,attn_layers=12,head_dim=256,cap=2048,vocab=248320,routed_b=3*48*10*2560*640/1e9),
 'glmflash':dict(active_b=18,layers=45,h=4096,heads=64,attn_layers=11,head_dim=256,cap=2048,vocab=154880,routed_b=3*42*8*4096*2048/1e9),
 'glm53':dict(active_b=40,layers=78,h=6144,heads=64,attn_layers=78,head_dim=256,cap=2048,vocab=154880,routed_b=3*75*8*6144*2048/1e9),
}
# Header-supported active matrix count for the audited Flash-Next target and MTP.
flash_dense=flash_mtp=0
for p in (ROOT/'data/weight-headers').glob('model*.header.json'):
    for name,t in json.loads(p.read_text()).items():
        if name=='__metadata__' or len(t['shape'])!=2 or not name.endswith('weight'):
            continue
        if 'mtp' in name:flash_mtp+=prod(t['shape'])
        elif 'visual' not in name and 'embed_tokens' not in name:flash_dense+=prod(t['shape'])


def target_components(model,context,rows):
    m=MODELS[model];n=m['active_b'];r=m['routed_b']
    if model=='flashnext':n=flash_dense/1e9+r
    ctx=min(context,m['cap']) if m['cap'] else context
    return [('target dense/shared/output',2*(n-r)*rows,'dense'),
            ('target routed experts',2*r*rows,'expert'),
            ('target attention QK/AV',4*m['attn_layers']*m['heads']*m['head_dim']*ctx*rows/1e9,'attention')]


def draft_components(kind,model,k,a,context,block=8):
    if kind=='none' or k==0:return []
    m=MODELS[model]
    if kind=='mtp':
        if model=='flashnext':
            params=flash_mtp/1e9+m['routed_b']/m['layers']+m['h']*m['vocab']/1e9
        else:
            # One target-sized layer plus shared output head and hidden/embedding fusion.
            head=m['h']*m['vocab']/1e9
            params=(m['active_b']-head)/m['layers']+head+2*m['h']**2/1e9
        ctx=min(context,m['cap']) if m['cap'] else context
        return [('MTP sequential proposals (layer + head)',2*params*k,'draft'),
                ('MTP attention',4*m['heads']*m['head_dim']*ctx*k/1e9,'attention')]
    d=json.loads((ROOT/'data/compute-evidence'/('dflash-glm53-config.json' if model=='glm53' else 'dflash-glmflash-config.json')).read_text())
    h=d['hidden_size'];q=d['num_attention_heads']*d['head_dim'];kv=d['num_key_value_heads']*d['head_dim'];l=d['num_hidden_layers']
    # Parallel drafter forward: do not multiply the whole block FLOPs by k again.
    body=l*(2*h*q+2*h*kv+3*h*d['intermediate_size'])
    query_rows=block-1
    window=min(context,d['sliding_window'])+query_rows
    fc=len(d['dflash_config']['target_layer_ids'])*m['h']*h
    return [('DFlash block body',2*body*query_rows/1e9,'draft'),
            ('DFlash full-vocabulary output head',2*h*d['vocab_size']*(block-1)/1e9,'draft'),
            ('DFlash target-feature projection',2*fc*a/1e9,'draft'),
            ('DFlash incremental context KV',4*l*h*kv*a/1e9,'draft'),
            ('DFlash attention',4*l*q*window*query_rows/1e9,'attention')]


def component_costs(parts,device,path):
    caps=INPUTS['devices'][device]['capacity_tops'];result=[]
    for name,ops,kind in parts:
        if ops==0:continue
        route=path
        if path=='fp4_mixed':route='fp4_matrix' if kind=='expert' else 'f16_matrix'
        elif path=='f32_simd' and (kind=='attention' or name.startswith('DFlash')):
            route='f16_matrix' # low-bit qmv alternative does not change the floating drafter
        elif kind in ('draft','attention') and path in ('int8_dot','int8_matrix'):
            route='f16_matrix' # separate floating draft / attention stage
        label={'f16_matrix':'FP16/BF16 operands, FP32 accumulate; matrix',
               'f16_simd':'FP16 operands; SIMD scenario',
               'f32_simd':'FP32 operands and accumulate; source-supported MLX qmv alternative',
               'fp4_matrix':'native NVFP4 operands; tensor',
               'int8_dot':'INT8 dot, INT32 accumulate; SIMD',
               'int8_matrix':'INT8 operands, INT32 accumulate; matrix'}[route]
        result.append(dict(stage=name,gops=ops,arithmetic=label,capacity_key=route,capacity_tops=caps[route],ideal_ms=ops/caps[route]))
    return result


def estimate(key,device,model,context,k,a,rate,workload_index=0):
    cfg=INPUTS['configurations'][key]
    candidates=[]
    ks=cfg.get('k_range',[k]);accepts=cfg.get('a_range',[a])
    # Raw Spark counters keep proposal/verification work and delivered decode tokens aligned.
    raw=None
    if key=='spark2':
        raw=json.loads((ROOT/'data/spark-decode-audit.json').read_text())['workloads'][workload_index]['runs']
    samples=[]
    if raw:
        for run in raw:
            count=run['spec']['drafts']
            samples.append((run['spec']['draft_tokens']/count,(run['completion_tokens']-1)/count,run['tok_s']))
    else:samples=[(kk,aa,rate) for kk in ks for aa in accepts if aa<=kk+1]
    for path in cfg['paths']:
        for kk,aa,rr in samples:
            blocks=[8] if cfg['draft_kind']=='dflash' else [0]
            if key=='m5ultra':blocks=[kk+1] # source adaptive blocks 4/8
            for block in blocks:
                parts=target_components(model,context,kk+1)
                draft=draft_components(cfg['draft_kind'],model,kk,aa,context,block)
                for draft_fraction in ([0,1] if key=='halo' else [1]):
                    # N-gram proposal work is host lookup, not a neural forward pass.
                    used=parts+[(n,x*draft_fraction,t) for n,x,t in draft]
                    costs=component_costs(used,device,path)
                    ms=sum(c['ideal_ms'] for c in costs)/aa
                    candidates.append(dict(path=path,k=kk,verify_rows=kk+1,emitted=aa,draft_block_rows=max(0,block-1),
                                           rate=rr,components=costs,ideal_ms_per_emitted=ms,
                                           demand=rr*ms/1000,compute_roof=1000/ms,
                                           target_gops_step=sum(x for _,x,_ in parts),
                                           draft_gops_step=sum(x*draft_fraction for _,x,_ in draft)))
    if not candidates:raise ValueError(f'No valid compute scenarios for {key}')
    # Spark raw workloads: medians over matched per-request calculations, not products of medians.
    if raw:
        ms=[median(c['ideal_ms_per_emitted'] for c in candidates)]
        demand=[median(c['demand'] for c in candidates)]
    else:
        ms=[min(c['ideal_ms_per_emitted'] for c in candidates),max(c['ideal_ms_per_emitted'] for c in candidates)]
        demand=[min(c['demand'] for c in candidates),max(c['demand'] for c in candidates)]
    return dict(ideal_ms_range=ms,demand_range=demand,roof_range=[1000/max(ms),1000/min(ms)],
                candidate_count=len(candidates),representative=candidates[len(candidates)//2],
                scenarios=candidates,note=cfg['note'],capacity_note=INPUTS['devices'][device]['note'],
                sample_basis='Matched raw requests; per-workload median' if raw else 'Conditional arithmetic/dispatch scenarios',
                scope='Target matrix/attention plus drafter arithmetic. Approximate active matrix counts except header-counted Flash-Next; excludes dequantization, selector/convolution, recurrence/rollback, padding and launches. DFlash full vocabulary head is a reference; optimized candidate heads can differ.')
