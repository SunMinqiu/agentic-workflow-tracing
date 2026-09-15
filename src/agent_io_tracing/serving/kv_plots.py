"""Create static plots from observed events and collector-side metric samples."""
from __future__ import annotations

from pathlib import Path
import json


def return_horizons(episodes, horizons=(1, 5, 10, 30)):
    result=[]
    for horizon in horizons:
        eligible=[e for e in episodes if e['observed_duration_s'] >= horizon or (e['status']=='returned' and e['removal_to_next_store_s'] <= horizon)]
        returned=sum(e['status']=='returned' and e['removal_to_next_store_s'] <= horizon for e in eligible)
        result.append({'horizon_s':horizon,'eligible_episodes':len(eligible),'returned_episodes':returned,'observed_return_fraction':returned/len(eligible) if eligible else None})
    return result


def write_plots(rep: Path, rows, episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory=rep/'visualizations'
    directory.mkdir(exist_ok=True)
    figures=[]
    metrics_path=rep/'kv_metrics.jsonl'
    metrics=[json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()] if metrics_path.is_file() else []
    if metrics:
        start=metrics[0]['client_epoch_s']
        x=[m['client_epoch_s']-start for m in metrics]
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
        axes[0].plot(x,[m['gauges'].get('vllm:kv_cache_usage_perc',float('nan'))*100 for m in metrics],color='#146b67')
        axes[0].set_ylabel('GPU cache use (%)')
        for key,label in [('vllm:num_requests_running','Running'),('vllm:num_requests_waiting','Waiting')]:
            axes[1].step(x,[m['gauges'].get(key,float('nan')) for m in metrics],where='post',label=label)
        axes[1].legend();axes[1].set_ylabel('Requests')
        removed=[(r['received_at_epoch_s']-start,len(r.get('block_hashes') or [])) for r in rows if r.get('event_type')=='BlockRemoved' and 'received_at_epoch_s' in r and r.get('phase')=='measurement']
        if removed:
            width=max(1,(max(x)-min(x))/60)
            edges=np.arange(0,max(max(x),max(t for t,_ in removed))+width*2,width)
            counts,_=np.histogram([t for t,_ in removed],bins=edges,weights=[n for _,n in removed])
            axes[2].bar(edges[:-1],counts/width,width=width,align='edge',color='#b87032')
        axes[2].set_ylabel('Received hash\nremovals / s');axes[2].set_xlabel('Seconds since first collector metric sample')
        axes[0].set_title('Cache pressure and event arrival rate')
        fig.suptitle('Event lane uses collector arrival time; it is not physical eviction timing.',fontsize=10)
        for ax in axes:ax.grid(alpha=.2)
        filename='kv_pressure.png';fig.savefig(directory/filename,dpi=140);plt.close(fig)
        figures.append({'path':'visualizations/'+filename,'title':'Cache pressure and received removal events'})
    if episodes:
        selected=[]
        for status,cpu in [('returned',True),('returned',False),('right_censored',False),('observation_gap',False),('reset',False)]:
            candidates=[e for e in episodes if e['status']==status and (bool(e['cpu_copy_continuously_observed'])==cpu if status=='returned' else True)]
            if candidates:selected.append(max(candidates,key=lambda e:e['observed_duration_s']))
        fig,ax=plt.subplots(figsize=(10,max(3,len(selected)*.7)),layout='constrained')
        for i,e in enumerate(selected):
            duration=e['observed_duration_s'];returned=e['status']=='returned'
            ax.plot([0,duration],[i,i],color='#146b67' if returned else '#b87032',linewidth=4)
            ax.scatter([0],[i],marker='x',color='black');ax.scatter([duration],[i],marker='o' if returned else '|',color='black')
        labels=[f"{str(e['block_hash'])[:12]} · g{e.get('group_idx')} · {e['status']}"+(' · CPU candidate' if e['cpu_copy_continuously_observed'] else '') for e in selected]
        ax.set_yticks(range(len(labels)),labels);ax.set_xlabel('Publisher seconds since each removal')
        ax.set_title('Representative logical removal episodes');ax.grid(axis='x',alpha=.2)
        filename='kv_episodes.png';fig.savefig(directory/filename,dpi=140);plt.close(fig)
        figures.append({'path':'visualizations/'+filename,'title':'Longest observed episode in each available status; CPU candidates are not confirmed loads'})
        horizons=return_horizons(episodes)
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        usable=[h for h in horizons if h['observed_return_fraction'] is not None]
        ax.bar(range(len(usable)),[h['observed_return_fraction']*100 for h in usable],color='#146b67')
        ax.set_xticks(range(len(usable)),[f"{h['horizon_s']} s\nn={h['eligible_episodes']}" for h in usable]);ax.set_ylim(0,105)
        ax.set_ylabel('Observed return fraction (%)');ax.set_title('Return within each horizon among observable episodes')
        fig.suptitle('Denominators vary by horizon. Episodes censored before that horizon are excluded.',fontsize=9)
        filename='kv_return_horizons.png';fig.savefig(directory/filename,dpi=140);plt.close(fig)
        figures.append({'path':'visualizations/'+filename,'title':'Fixed-horizon observations, not an estimate for all removed blocks'})
    else:
        horizons=[]
    return figures,horizons


def write_reload_plot(rep, rows, transfers):
    jobs=[row for row in transfers if row['confirmed'] and row['direction']=='load']
    if not jobs:
        return None
    job=min(jobs,key=lambda row:row['all_workers_observed_at_ns'])
    completed=[row for row in rows if row['event_type']=='TransferCompleted' and row.get('associated_request_id')==job['request_id'] and row.get('job_id')==job['job_id']]
    completed=list({(r['publisher_epoch'],r['sequence']):r for r in completed}.values())
    completed.sort(key=lambda row:row['process_id'])
    if not completed or any('submitted_at_ns' not in row for row in completed):
        return None
    request=[row for row in rows if row.get('request_id')==job['request_id']]
    blocked=next((r for r in request if r['event_type']=='RequestLoadBlocked'),None)
    unblocked=next((r for r in request if r['event_type']=='RequestLoadUnblocked' and blocked and r['time_monotonic_ns']>blocked['time_monotonic_ns']),None)
    start=min([r['submitted_at_ns'] for r in completed]+([blocked['time_monotonic_ns']] if blocked else []))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    labels=[];right=0
    for index,row in enumerate(completed):
        lo=(row['submitted_at_ns']-start)/1e6;hi=(row['time_monotonic_ns']-start)/1e6
        ax.barh(index,hi-lo,left=lo,height=.5,color='#146b67')
        ax.text(hi+.15,index,f"reported transfer: {row['transfer_time_s']*1000:.2f} ms",va='center',fontsize=9)
        labels.append(f"Worker PID {row['process_id']}");right=max(right,hi)
    if blocked and unblocked:
        lo=(blocked['time_monotonic_ns']-start)/1e6;hi=(unblocked['time_monotonic_ns']-start)/1e6
        ax.barh(len(labels),hi-lo,left=lo,height=.5,color='#b87032')
        ax.text(hi+.15,len(labels),f'{hi-lo:.2f} ms',va='center',fontsize=9)
        labels.append('Observed request wait');right=max(right,hi)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,right*1.7+1)
    ax.set_xlabel('Monotonic milliseconds since the earliest shown event')
    ax.set_title('First confirmed CPU → GPU job in this run')
    ax.grid(axis='x',alpha=.2)
    fig.suptitle('Worker bars span submission to observed completion; printed transfer times are separate measurements.',fontsize=9)
    directory=rep/'visualizations';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'kv_confirmed_reload.png',dpi=140);plt.close(fig)
    return {'path':'visualizations/kv_confirmed_reload.png','title':'One confirmed load: worker completion observations and request wait','source_request_id':job['source_request_id'],'job_id':job['job_id']}


def write_eviction_plot(rep, evictions):
    values=sorted(r['time_since_last_release_s'] for r in evictions if r.get('time_since_last_release_s') is not None)
    if not values:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.step(values,[(i+1)/len(values)*100 for i in range(len(values))],where='post',color='#146b67')
    ax.set_xlabel('Seconds from reference count reaching zero to observed capacity eviction')
    ax.set_ylabel('Cumulative fraction of observed evictions (%)')
    ax.set_title(f'Idle interval before cached-content eviction · n={len(values):,}')
    ax.grid(alpha=.2);ax.set_ylim(0,102)
    directory=rep/'visualizations';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'kv_physical_idle.png',dpi=140);plt.close(fig)
    return {'path':'visualizations/kv_physical_idle.png','title':'Physical cached-content evictions: time since last release, observed events only'}
