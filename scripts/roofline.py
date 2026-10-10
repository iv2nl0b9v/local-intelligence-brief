"""Comparative decode rooflines: explicit assumptions, sensitivity, and profile evidence.
Demand ratios are not time fractions. Compute sums target and drafter costs at
stage-specific precision capacities. Forecasts are not validation.
"""
from pathlib import Path
from html import escape
from math import log10
import json, re, sys
import compute_roofline as compute
ROOT = Path(__file__).resolve().parents[1]
INPUTS = json.loads((ROOT/'data/roofline-inputs.json').read_text(encoding='utf-8'))
DEVICES, MODELS = INPUTS['DEVICES'], INPUTS['MODELS']


def rng(v, fmt=lambda x:f'{x:.1f}'):
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f'{lo}–{hi}'


def pct(v):
    return rng(v,lambda x:f'{100*x:.0f}%')


def step(device, model, d, ctx, moe, k, a, expert_fraction=.5, union=None):
    n,layers,heads,hd,cap,cache_per_token,index_per_token = MODELS[model]
    bw,vec,matrix = DEVICES[device][1:4]
    attended = min(ctx,cap) if cap else ctx
    # Explicit dense/routed split: dense weights once, selected expert union once.
    # Default central scenario recovers the old 1+.25k multiplier, with no claim
    # that .5 routed fraction or 1+.5k union are observations.
    union = 1+.5*k if union is None else union
    weights = d*((1-expert_fraction)+expert_fraction*union) if moe else d
    cache = (cache_per_token*attended+index_per_token*ctx)/1e9
    gb = (weights+cache)/a  # optimistic cache reuse across verify rows
    return dict(gb_per_token=gb,bandwidth_roof=bw/gb,memory_ms=1000*gb/bw,
                context=ctx,draft_tokens=k,emitted_tokens=a,
                weight_gb_per_step=weights,cache_gb_per_step=cache)



def analyse(p, canonical=None):
    key,device,model,d,speed,ctx,moe,k,a,estimate,note=p
    if canonical:
        speed,estimate,note=canonical['speed'],canonical['estimate'],canonical['note']
    ks = k if isinstance(k,list) else [k,k]
    ends=[step(device,model,d,c,moe,kk,aa) for c,kk,aa in zip(ctx,ks,a)]
    mem=[s/e['bandwidth_roof'] for s,e in zip(speed,ends)]
    calculations=[compute.estimate(key,device,model,c,kk,aa,rr,i)
                  for i,(c,kk,aa,rr) in enumerate(zip(ctx,ks,a,speed))]
    comp=[x for result in calculations for x in result['demand_range']]
    for e,result in zip(ends,calculations):e['compute']=result
    # Do not clip contradictory >100% demand: that signals invalid assumptions.
    gap=[1-max(m,c) for m,result in zip(mem,calculations) for c in result['demand_range']]
    sensitivity=[]
    for s,c,kk,aa in zip(speed,ctx,ks,a):
        # Exact endpoint for audited weights, otherwise +/-25% traffic scenario.
        scales=[1] if key=='pro6000' else [.75,1.25]
        candidates=[step(device,model,d*scale,c,moe,kk,aa,f,u)
                    for scale in scales for f in ([.25,.75] if moe and kk else [.5])
                    for u in ([1,kk+1] if moe and kk else [1])]
        demand=[s/e['bandwidth_roof'] for e in candidates]
        sensitivity.extend(demand)
    ratio=min(min(result['roof_range'])/e['bandwidth_roof'] for result,e in zip(calculations,ends))
    limiting='Bandwidth-led scenarios' if ratio>2 else 'Compute / bandwidth sensitive'

    r=dict(device=DEVICES[device][0],model=model,estimate=estimate,note=note,speed=list(speed),
           display=canonical['display'] if canonical else rng(speed),endpoints=ends,
           bandwidth_attainment=mem,compute_attainment=comp,roof_gap=gap,
           bandwidth_demand_sensitivity=[min(sensitivity),max(sensitivity)],
           limiting_reference=limiting,compute_to_bw_roof_ratio=ratio,
           assumption_note=INPUTS['notes'].get(key,INPUTS['notes']['default']),
           profile_status='Related EXL3 profile: bandwidth-dominated kernels' if key=='spark2' else 'No matching profile',
           inputs=dict(weight_gb_per_pass=d,context=ctx,k=ks,a=a,moe=moe,device=device),
           scenario_only=bool(estimate))
    if canonical and 'scenario' in canonical:r['scenario']=canonical['scenario']
    return r


def build():
    canonical=json.loads((ROOT/'data/configurations.json').read_text(encoding='utf-8'))
    rows={p[0]:analyse(p,canonical[p[0]]) for p in INPUTS['PAIRS']}
    refs={p[0]:analyse(p) for p in INPUTS['REFERENCES']}
    profiles=json.loads((ROOT/'data/spark-profile.json').read_text())
    return rows,refs,profiles


def cells(key,r):
    hint=escape(r['limiting_reference']+'; '+r['profile_status']+'; assumptions and sensitivity in Sources & calculations',quote=True)
    cls=' hot mem' if r['limiting_reference']=='Bandwidth-led scenarios' else ' hot cmp'
    return (f'<td data-roofline="{key}" class="tt{cls}" title="{hint}">{pct(r["bandwidth_attainment"])}</td>'
            f'<td data-roofline-part class="tt" title="Precision-specific target + drafter arithmetic demand; scenario range, not measured utilization">{rng(r["compute_attainment"],lambda v:f"{100*v:.1f}%")}</td>'
            f'<td data-roofline-part class="tt" title="Gap to modeled peak; not measured inference overhead">{pct(r["roof_gap"])}</td>')


def chart(rows):
    w,h,ml,mr,mt,mb=760,410,65,20,25,55
    px=lambda v:ml+(log10(max(v,1e-4))+4)/5*(w-ml-mr)
    py=lambda v:mt+(1-log10(max(v,1e-4)))/5*(h-mt-mb)
    out=[f'<svg class="roofline-chart" viewBox="0 0 {w} {h}" role="img" aria-labelledby="rf-title rf-desc"><title id="rf-title">Restored comparative decode roofline</title><desc id="rf-desc">Normalized reference model, not measured kernel rooflines. Blue points are reported speeds, hollow points forecasts. Lines connect workload endpoints. Diagonal is modeled bandwidth, horizontal is the precision-specific compute roof. Mixed-precision operation costs are summed; this is a normalized time-roof comparison.</desc>']
    for e in range(-4,2):
        x,y=px(10**e),py(10**e)
        out.append(f'<line class="rf-grid" x1="{x}" x2="{x}" y1="{mt}" y2="{h-mb}"/><text class="rf-tick" x="{x}" y="{h-mb+17}" text-anchor="middle">{10**e:g}×</text>')
        out.append(f'<line class="rf-grid" x1="{ml}" x2="{w-mr}" y1="{y}" y2="{y}"/><text class="rf-tick" x="{ml-8}" y="{y+4}" text-anchor="end">{10**e*100:g}%</text>')
    out.append(f'<polyline class="rf-roof" points="{px(.0001)},{py(.0001)} {px(1)},{py(1)} {px(10)},{py(1)}"/>')
    out.append(f'<text class="rf-label" x="{px(.0004)}" y="{py(.004)}">bandwidth reference</text><text class="rf-label" x="{px(1.1)}" y="{py(1)-9}">precision-specific compute</text>')
    for key,r in rows.items():
        points=[]
        for e,m in zip(r['endpoints'],r['bandwidth_attainment']):
            for c in e['compute']['demand_range']:
                points.append((px(c/m),py(c)))
        out.append(f'<line x1="{points[0][0]}" y1="{points[0][1]}" x2="{points[-1][0]}" y2="{points[-1][1]}" stroke="#2a78d6"/>')
        for x,y in points:
            fill='none' if r['estimate'] else '#2a78d6'
            out.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{fill}" stroke="#2a78d6" stroke-width="2"><title>{escape(r["device"])}: {r["display"]} tok/s; BW demand {pct(r["bandwidth_attainment"])}; {r["limiting_reference"]}</title></circle>')
        if key in ['pro6000','spark2','rtx4090','m5pro']:
            x,y=points[-1];dy={'spark2':-17,'rtx4090':20}.get(key,-7)
            out.append(f'<text class="rf-label" x="{x+8}" y="{y+dy}">{escape(key)}</text>')
    out.append(f'<text class="rf-axis" x="400" y="{h-8}" text-anchor="middle">Ideal compute time ÷ ideal bandwidth time (log)</text><text class="rf-axis" x="15" y="200" transform="rotate(-90 15 200)" text-anchor="middle">Precision-specific compute demand (log)</text></svg>')
    return '<p>Solid: reported rate. Hollow: forecast. Connectors span workloads and explicit kernel-path scenarios, not confidence intervals. Hover for pairing details.</p>'+''.join(out)


def table(rows,refs):
    out=['<h4>Precision-specific compute ledger</h4><p>Each endpoint lists target and drafter work, verification/proposal counts, and the actual or conditional arithmetic path. GOps count multiply and add separately; integer dot work is GOP/TOPS, not floating FLOPs. Scenario envelopes are not confidence intervals. Per-component records and every scenario are exported in data/roofline.json.</p><div style="overflow-x:auto"><table class="method-table"><thead><tr><th>Pairing / endpoint</th><th>Verify / emitted / drafter</th><th>Target / draft GOps per step</th><th>Compute path / capacity</th><th>Compute ms per emitted token / roof tok/s</th><th>BW / compute demand</th><th>Evidence &amp; limits</th></tr></thead><tbody>']
    for key,r in {**rows,**refs}.items():
        for i,e in enumerate(r['endpoints']):
            c=e['compute'];sc=c['scenarios'];q=c['representative']
            paths=sorted({(x['arithmetic'],x['capacity_tops']) for v in sc for x in v['components']})
            path='<br>'.join(escape(label)+f': {cap:g} TOp/s' for label,cap in paths)
            out.append(f'<tr><td><b>{escape(r["device"])}</b><br>{escape(r["note"])}<br>endpoint {i+1}: {r["speed"][i]:.2f} tok/s</td>'
                       f'<td>verify {rng([v["verify_rows"] for v in sc])}<br>emit {rng([v["emitted"] for v in sc])}<br>draft rows {rng([v["draft_block_rows"] or v["k"] for v in sc])}</td>'
                       f'<td>target {rng([v["target_gops_step"] for v in sc])}<br>draft {rng([v["draft_gops_step"] for v in sc])}</td>'
                       f'<td>{path}</td><td>{rng(c["ideal_ms_range"],lambda x:f"{x:.3f}")} ms<br>{rng(c["roof_range"],lambda x:f"{x:.0f}")} tok/s</td>'
                       f'<td>BW {100*r["bandwidth_attainment"][i]:.1f}%<br>compute {rng(c["demand_range"],lambda x:f"{100*x:.2f}%")}</td>'
                       f'<td>{escape(c["note"])}<br>{escape(c["capacity_note"])}<br>{escape(c["sample_basis"])}. {escape(c["scope"])}</td></tr>')
    out.append('</tbody></table></div><h4>Bandwidth assumptions and sensitivity retained</h4><table class="method-table"><thead><tr><th>Pairing</th><th>GB / emitted token</th><th>BW roof tok/s</th><th>BW demand sensitivity</th><th>Interpretation</th></tr></thead><tbody>')
    for r in {**rows,**refs}.values():
        es=r['endpoints'];extra=' Forecast: self-consistency only.' if r['estimate'] else ''
        out.append(f'<tr><td>{escape(r["device"])}</td><td>{rng([e["gb_per_token"] for e in es])}</td><td>{rng([e["bandwidth_roof"] for e in es],lambda v:f"{v:.0f}")}</td><td>{pct(r["bandwidth_demand_sensitivity"])}</td><td>{escape(r["limiting_reference"]+extra+" "+r["assumption_note"])}</td></tr>')
    out.append('</tbody></table><h4>Compute-capacity provenance</h4><table class="method-table"><thead><tr><th>Device</th><th>Dense precision capacities (TOp/s)</th><th>Source / qualification</th></tr></thead><tbody>')
    for key,d in compute.INPUTS['devices'].items():
        out.append(f'<tr><td>{escape(key)}</td><td>{escape(str(d["capacity_tops"]))}</td><td>{escape(d["note"])} <a href="{d["source"]}">Source</a></td></tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def profile_table(profile):
    out=['<h4>Measured Spark step anatomy — separate from the roofline estimates</h4>',
         f'<p>{escape(profile["snapshot"])}. {escape(profile["classification"])}. <a href="{profile["source"]}">Pinned source data</a>.</p>',
         '<table class="method-table"><thead><tr><th>Rank / verify rows</th><th>Step ms</th><th>BW-class kernels ms / share</th><th>Collectives ms</th><th>Small kernels ms</th><th>Idle ms / share</th></tr></thead><tbody>']
    for r in profile['rows']:
        out.append(f'<tr><td>{r["rank"]:.0f} / {r["M_verify_rows"]:.0f}</td><td>{r["wall_ms"]:.2f}</td><td>{r["bandwidth_ms"]:.2f} / {r["bandwidth_ms"]/r["wall_ms"]:.1%}</td><td>{r["nccl_ms"]:.2f}</td><td>{r["small_ms"]:.2f}</td><td>{r["idle_ms"]:.2f} / {r["idle_ms"]/r["wall_ms"]:.1%}</td></tr>')
    out.append('</tbody></table><p>Rank-0 M=6: removing all 3.23 ms idle would improve the 75.09 ms step by only 4.5% at unchanged acceptance. Removing all 13.17 ms outside the bandwidth-class kernels is an optimistic 21.3% speedup. Neither is an achievable promise: work can overlap, classifications are approximate, and essential collectives cannot simply disappear.</p>')
    return ''.join(out)


def render(html,rows,refs,pre):
    block=chart(rows)+profile_table(pre)+table(rows,refs)
    html=re.sub(r'(<!-- roofline:start -->).*?(<!-- roofline:end -->)',lambda m:m[1]+block+m[2],html,flags=re.S)
    for key,r in rows.items():
        html,count=re.subn(rf'<td data-roofline="{key}"[^>]*>.*?</td>(?:<td data-roofline-part[^>]*>.*?</td>)*',lambda m:cells(key,r),html,flags=re.S)
        if count!=1:raise ValueError(f'Expected one evidence cell for {key}; found {count}')
    return html


def serialized(rows,refs,pre):
    return json.dumps({'method':'Comparative decode roofline scenarios, not elapsed-time attribution. Traffic sensitivity is illustrative, not a confidence interval.','pairs':rows,'references':refs,'devices':DEVICES,'compute_inputs':compute.INPUTS,'profile':pre},indent=2,ensure_ascii=False)+'\n'


def main():
    result=build();path=ROOT/'index.html';html=path.read_text(encoding='utf-8')
    new,data=render(html,*result),serialized(*result);target=ROOT/'data/roofline.json'
    if '--check' in sys.argv:
        if new!=html or target.read_text(encoding='utf-8')!=data:raise SystemExit('Validation failed: stale roofline content')
        print('Roofline content matches; no files changed.')
    else:
        path.write_text(new,encoding='utf-8',newline='\n');target.write_text(data,encoding='utf-8',newline='\n')
        print('Updated bandwidth and precision-specific compute rooflines.')
if __name__=='__main__':main()
