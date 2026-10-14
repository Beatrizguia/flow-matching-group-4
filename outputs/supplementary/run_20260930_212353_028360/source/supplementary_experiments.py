"""Fixed, descriptive extension: validation ablation and test training-seed robustness.
No hyperparameter selection from these experiments. Main torus protocol is unchanged.
"""
from pathlib import Path
from datetime import datetime
import json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_profile_project import (Velocity, prepare_data, train_model, evaluate,
                                  sample_four, constraint_matrix, metrics)

ROOT = Path(__file__).resolve().parent

def paired_table(frame, metric='sliced_w1'):
    pivot = frame.pivot(index='seed', columns='method', values=metric)
    return pd.DataFrame({m+'-A': pivot[m]-pivot['A'] for m in 'BCD'})


def load_model(run):
    cp = torch.load(run/'model.pt', map_location='cpu', weights_only=False)
    model = Velocity()
    model.load_state_dict(cp['model_state_dict'])
    return model.eval()


def main():
    torch.set_num_threads(4)
    arrays, info = prepare_data(ROOT/'fluvius_wide_format.csv')
    base = ROOT/json.loads((ROOT/'load_run.json').read_text())['run_dir']
    out = ROOT/'outputs'/'supplementary'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    # Written before experiments: fixed settings, no winner selection.
    config = dict(training_seeds=[42,43,44], sampling_seeds=list(range(10)),
                  training_iterations=5000, n_samples=2000, n_steps=100,
                  ablation_split='validation', robustness_split='test',
                  ablation_model_training_seed=42,
                  constraints=['nonnegative','ramp','both'], soft_strength=.3,
                  soft_divisor=9, sw_seed=2026, sw_directions=128,
                  selection='No tuning; descriptive follow-up after historical test inspection',
                  base_run=str(base.relative_to(ROOT)), data_sha256=info['sha256'],
                  versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','matplotlib']},
                  python=platform.python_version(),
                  code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ['load_profile_project.py','supplementary_experiments.py']})
    (out/'protocol.json').write_text(json.dumps(config,indent=2))
    (out/'data_config.json').write_text(json.dumps(info,indent=2))
    model = load_model(base)
    scale, ramp = info['scale'], info['ramp_threshold']
    dirs = np.random.default_rng(2026).normal(size=(24,128));dirs /= np.linalg.norm(dirs,axis=0)
    rows, traces = [], []
    for constraints in config['constraints']:
        for seed in config['sampling_seeds']:
            noise = torch.randn(2000,24,generator=torch.Generator().manual_seed(seed))
            idx = np.random.default_rng(10000+seed).choice(len(arrays['validation']),2000,replace=False)
            ref = arrays['validation'][idx]
            def record(t, method, state, correction, event):
                G,b = constraint_matrix(ramp/scale,state.device,state.dtype,constraints)
                traces.append(dict(constraints=constraints,seed=seed,time=t,method=method,event=event,
                                   mean_consumption=float(state.mean())*scale,
                                   mean_active_violation=float(torch.relu(state@G.T-b).mean())*scale,
                                   correction_l2=float(torch.linalg.vector_norm(correction,dim=1).mean())*scale))
            for method in 'ABCD':record(0.,method,noise,torch.zeros_like(noise),'initial')
            samples = sample_four(model,noise,ramp/scale,constraints=constraints,trace=record)
            saved = {}
            for method, value in samples.items():
                x = value.numpy()*scale;saved[method]=x
                m = metrics(x,ref,ramp,dirs,max(1e-5,scale*1e-5))
                G,b = constraint_matrix(ramp,'cpu',torch.float32,constraints)
                m['active_feasible_fraction'] = float((torch.relu(torch.from_numpy(x)@G.T-b).max(1).values<=1e-5).float().mean())
                m['mean_consumption'] = float(x.mean())
                rows.append(dict(constraints=constraints,seed=seed,method=method,**m))
            if seed==0:np.savez_compressed(out/f'ablation_{constraints}_seed_0.npz',reference=ref,**saved)
        print('Finished validation ablation:',constraints,flush=True)
    ablation = pd.DataFrame(rows);ablation.to_csv(out/'ablation_metrics.csv',index=False)
    ablation.groupby(['constraints','method'])[['active_feasible_fraction','feasible_fraction','sliced_w1','mean_consumption']].agg(['mean','std']).to_csv(out/'ablation_summary.csv')
    trace = pd.DataFrame(traces);trace.to_csv(out/'trajectories.csv',index=False)
    history_rows = []
    # Training interpolation diagnostic uses the same normalized Gaussian and real training endpoints.
    rng=np.random.default_rng(9001)
    endpoint=torch.from_numpy(arrays['train'][rng.choice(len(arrays['train']),2000,replace=False)]/scale)
    source=torch.randn(2000,24,generator=torch.Generator().manual_seed(9001))
    G,b=constraint_matrix(ramp/scale,'cpu',torch.float32)
    for t in np.linspace(0,1,101):
        x=(1-t)*source+t*endpoint
        history_rows.append(dict(time=t,mean_consumption=float(x.mean())*scale,
                                 mean_active_violation=float(torch.relu(x@G.T-b).mean())*scale))
    pd.DataFrame(history_rows).to_csv(out/'training_path_diagnostic.csv',index=False)
    plot_supplement(out, ablation, trace, pd.DataFrame(history_rows))
    # Three training seeds, each summarized over ten common sampling/reference seeds.
    all_frames=[]
    historical=pd.read_csv(base/'metrics.csv');historical['training_seed']=42;all_frames.append(historical)
    for seed in [43,44]:
        run=out/f'training_seed_{seed}';run.mkdir()
        trained,history,selected=train_model(arrays['train'],arrays['validation'],scale,run,seed=seed)
        frame,_,_=evaluate(trained,arrays,info,run)
        frame['training_seed']=seed;all_frames.append(frame)
        (run/'config.json').write_text(json.dumps(dict(training_seed=seed,selected_step=selected,
                                                     protocol='../protocol.json'),indent=2))
    multi=pd.concat(all_frames,ignore_index=True);multi.to_csv(out/'training_seed_metrics.csv',index=False)
    per_training=multi.groupby(['training_seed','method'])[['feasible_fraction','sliced_w1','daily_total_w1']].mean()
    per_training.to_csv(out/'per_training_seed.csv')
    per_training.groupby('method').agg(['mean','std']).to_csv(out/'between_training_summary.csv')
    for name,frame in [('historical',historical)]:
        paired=paired_table(frame);paired.to_csv(out/f'{name}_paired_sw1.csv')
        paired.agg(['mean','std']).T.to_csv(out/f'{name}_paired_sw1_summary.csv')
    paired=multi.pivot(index=['training_seed','seed'],columns='method',values='sliced_w1')
    delta=pd.DataFrame({m+'-A':paired[m]-paired.A for m in 'BCD'})
    delta.to_csv(out/'paired_sw1_by_training_seed.csv')
    delta.groupby('training_seed').mean().to_csv(out/'paired_training_means.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    means=per_training.reset_index()
    for method,g in means.groupby('method'):
        axes[0].plot(g.training_seed,g.sliced_w1,'o-',label=method)
        axes[1].plot(g.training_seed,g.feasible_fraction,'o-',label=method)
    axes[0].set(ylabel='SW1 (CSV units)');axes[1].set(ylabel='Feasible fraction')
    for ax in axes:ax.set(xlabel='Training seed',xticks=[42,43,44]);ax.legend()
    fig.suptitle('Each point averages 10 sampling seeds; 3 trained models')
    fig.tight_layout();fig.savefig(out/'training_robustness.png',dpi=160);plt.close(fig)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'complete','rows':len(multi)},indent=2))
    (ROOT/'supplementary_run.json').write_text(json.dumps({'run_dir':str(out.relative_to(ROOT))},indent=2)+'\n')
    print('COMPLETE',out,flush=True)


def plot_supplement(out, ablation, trace, interpolation):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    selected=trace[trace.constraints=='both']
    for method in 'ABCD':
        g=selected[selected.method==method].groupby(['time','event'],sort=False)[['mean_consumption','mean_active_violation','correction_l2']].mean().reset_index()
        for ax,key in zip(axes,['mean_consumption','mean_active_violation','correction_l2']):ax.plot(g.time,g[key],label=method,alpha=.85)
    for ax,key in zip(axes[:2],['mean_consumption','mean_active_violation']):
        ax.plot(interpolation.time,interpolation[key],'k--',label='Training interpolation')
    for ax,label in zip(axes,['Mean consumption','Mean active violation','Mean correction L2']):
        ax.set(xlabel='Flow time',ylabel=label+' (CSV units)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.suptitle('Both constraints: means over 10 sampling seeds; D projects only at t=1')
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,title in zip(axes,['sliced_w1','active_feasible_fraction','feasible_fraction'],['Validation SW1','Feasibility: active constraints','Feasibility: both constraints']):
        grouped=ablation.groupby(['constraints','method'])[key].agg(['mean','std'])
        for i,c in enumerate(['nonnegative','ramp','both']):
            g=grouped.loc[c].reindex(list('ABCD'));ax.bar(np.arange(4)+(i-1)*.25,g['mean'],.25,yerr=g['std'],capsize=2,label=c)
        ax.set(xticks=np.arange(4),xticklabels=list('ABCD'),title=title)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle('Fixed model (training seed 42), 10 paired sampling seeds; mean +/- SD')
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(out/'ablation.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
