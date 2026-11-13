#!/usr/bin/env python3
"""
STUDY 09 — AMPLITUDE-ENCODING GEOMETRY: SIGN, NORM, AND ANCHOR TESTS
===================================================================

Purpose
-------
Study 08 found that amplitude fidelity is squared cosine similarity and that
centering can create negative cosine pairs. Study 09 isolates the mechanisms:

A. Sign ambiguity: compare signed cosine, absolute cosine, and squared cosine.
   The last is exactly the fidelity produced by real amplitude encoding.
B. Norm loss: test whether pre-normalization vector norm carries class and
   neighbour information that amplitude normalization discards.
C. Anchor augmentation: append a fixed reference amplitude to each centered
   feature vector before normalization. This changes the state geometry and
   tests whether it reduces sign ambiguity / improves Euclidean-neighbour
   recovery. It is an experimental encoding variant, not a proven solution.
D. Dataset structure: include real datasets and controlled synthetic data
   with directional or magnitude-dominated class structure.

No quantum hardware, noise model, or shot simulation is used here. All scores
are exact statevector inner products / exact classical diagnostic metrics.
The fixed-anchor state is a valid normalized real amplitude vector, but its
preparation and resource costs are not modelled.

Dependencies:
  pip install numpy pandas scipy scikit-learn openpyxl

Quick test:
  python 09_amplitude_geometry_mechanisms.py --quick

Recommended full run:
  python 09_amplitude_geometry_mechanisms.py \
    --datasets iris wine breast_cancer digits synthetic_directional synthetic_magnitude \
    --n-splits 20 --max-train 120 --max-test 50 --max-features 8 \
    --k-values 1 3 5 9 --anchor-strengths 0.25 0.5 1 2 \
    --outdir results/study_09 --workbook study_09_amplitude_geometry.xlsx

Output: one Excel workbook with separate worksheets for all result tables.
"""
from __future__ import annotations
import argparse, os, math, warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f_oneway
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                              load_digits, make_blobs)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

EPS = 1e-12


def load_data(name: str, seed: int):
    if name == 'iris':
        d=load_iris(); return d.data, d.target
    if name == 'wine':
        d=load_wine(); return d.data, d.target
    if name == 'breast_cancer':
        d=load_breast_cancer(); return d.data, d.target
    if name == 'digits':
        d=load_digits(); return d.data, d.target
    if name == 'synthetic_directional':
        # Classes are primarily separated by direction; norm is deliberately
        # randomized independently to avoid making magnitude a shortcut.
        rng=np.random.default_rng(5101)
        n=900; d=8; y=np.repeat(np.arange(3), n//3)
        centers=np.array([[1,0,0,0,0,0,0,0],[-.5,.866,0,0,0,0,0,0],[-.5,-.866,0,0,0,0,0,0]])
        X=np.vstack([centers[c] + rng.normal(0,.28,d) for c in y])
        scales=rng.uniform(.5,2.0,n)
        X=X*scales[:,None]
        return X,y
    if name == 'synthetic_magnitude':
        # Direction is shared; classes differ primarily in radial magnitude.
        rng=np.random.default_rng(5102)
        n=900; d=8; y=np.repeat(np.arange(3), n//3)
        # Positive near-collinear vectors at three radial scales. Per-feature
        # MinMax scaling keeps the classes radially ordered, while amplitude
        # normalization maps nearly collinear vectors to nearly identical
        # states: a deliberately controlled test of magnitude loss.
        direction=np.arange(1,d+1,dtype=float); direction/=np.linalg.norm(direction)
        radii=np.choose(y,[0.45,1.0,2.0]) * rng.lognormal(0,.035,n)
        X=radii[:,None]*direction[None,:] + rng.normal(0,.002,(n,d))
        return X,y
    raise ValueError(f'Unknown dataset: {name}')


def cap_stratified(X,y,max_n,seed):
    if max_n is None or len(y)<=max_n: return X,y
    a,_,b,_=train_test_split(X,y,train_size=max_n,stratify=y,random_state=seed)
    return a,b


def normalize_rows(X):
    norms=np.linalg.norm(X,axis=1)
    out=X/np.maximum(norms[:,None],EPS)
    out[norms<EPS]=0.0
    return out,norms


def sim_cos(A,B):
    An,_=normalize_rows(A); Bn,_=normalize_rows(B)
    return np.clip(An@Bn.T,-1,1)


def sim_euclidean(A,B):
    return -np.linalg.norm(A[:,None,:]-B[None,:,:],axis=2)


def topk(row,k):
    # Stable deterministic tie handling for the geometry-level comparison.
    return np.argsort(-row,kind='mergesort')[:min(k,len(row))]


def jaccard(a,b):
    a=set(map(int,a)); b=set(map(int,b)); return len(a&b)/max(1,len(a|b))


def spearman_safe(a,b):
    if len(a)<3 or np.std(a)<EPS or np.std(b)<EPS: return np.nan
    v=spearmanr(a,b).statistic
    return float(v) if np.isfinite(v) else np.nan


def knn_predict(S,ytr,k):
    preds=[]
    for row in S:
        ix=topk(row,k); vals,cnt=np.unique(ytr[ix],return_counts=True)
        preds.append(vals[np.argmax(cnt)])
    return np.asarray(preds)


def eta_squared(values, labels):
    values=np.asarray(values); labels=np.asarray(labels)
    grand=np.mean(values); ss_total=np.sum((values-grand)**2)
    if ss_total<EPS: return 0.0
    ss_between=0.0
    for c in np.unique(labels):
        v=values[labels==c]
        ss_between += len(v)*(np.mean(v)-grand)**2
    return float(ss_between/ss_total)


def build_variants(Xtr, Xte, anchor_strengths):
    """Feature arrays whose normalized inner products define amplitude states."""
    mu=Xtr.mean(axis=0)
    variants={
      'minmax_raw':(Xtr,Xte,'raw_minmax',np.nan),
      'center_0.5':(Xtr-.5,Xte-.5,'fixed_0.5',np.nan),
      'center_train_mean':(Xtr-mu,Xte-mu,'train_mean',np.nan),
    }
    # Anchor is a constant reference coordinate. It prevents arbitrary sign
    # reversal from leaving fidelity unchanged once dot products are offset.
    for a in anchor_strengths:
        variants[f'anchor_train_mean_{a:g}']=(
            np.column_stack([Xtr-mu,np.full(len(Xtr),a)]),
            np.column_stack([Xte-mu,np.full(len(Xte),a)]),
            'train_mean_plus_fixed_anchor',float(a))
    return variants,mu


def run(args):
    rows=[]; split_rows=[]; pair_rows=[]; norm_rows=[]; pairdiag=[]; anchor_rows=[]
    dataset_rows=[]
    for ds_idx,name in enumerate(args.datasets):
        print(f'\nDataset: {name}',flush=True)
        X,y=load_data(name,args.seed)
        sss=StratifiedShuffleSplit(n_splits=args.n_splits,test_size=args.test_size,random_state=args.seed+ds_idx)
        for split,(itr,ite) in enumerate(sss.split(X,y)):
            Xtr0,ytr=X[itr],y[itr]; Xte0,yte=X[ite],y[ite]
            Xtr0,ytr=cap_stratified(Xtr0,ytr,args.max_train,args.seed+split)
            Xte0,yte=cap_stratified(Xte0,yte,args.max_test,args.seed+1000+split)
            # Fit scaler on training data only; no test leakage.
            scaler=MinMaxScaler()
            Xtr_all=scaler.fit_transform(Xtr0); Xte_all=np.clip(scaler.transform(Xte0),0,1)
            # Train-only unsupervised variance selection; same columns for all maps.
            vari=np.var(Xtr_all,axis=0)
            chosen=np.argsort(-vari,kind='mergesort')[:min(args.max_features,Xtr_all.shape[1])]
            Xtr=Xtr_all[:,chosen]; Xte=Xte_all[:,chosen]
            ntr,nte=len(ytr),len(yte)
            S_euc=sim_euclidean(Xte,Xtr)
            S_cos=sim_cos(Xte,Xtr)
            for k in args.k_values:
                ref=knn_predict(S_euc,ytr,k)
                ref_neigh=[topk(r,k) for r in S_euc]
                se=np.sort(S_euc,axis=1)[:,::-1]
                rows.append({'dataset':name,'split':split,'variant':'classical_euclidean','metric':'negative_euclidean',
                  'k':k,'accuracy':float(np.mean(ref==yte)),'neighbour_jaccard_vs_euclidean':1.0,
                  'prediction_disagreement_vs_euclidean':0.0,'mean_rank_spearman_vs_euclidean':1.0,
                  'mean_boundary_gap':float(np.mean(se[:,k-1]-se[:,k])) if ntr>k else np.nan})
                # Classical cosine diagnostic; signed cosine is not itself a
                # SWAP-test fidelity estimate for real amplitude states.
                p=knn_predict(S_cos,ytr,k); neigh=[topk(r,k) for r in S_cos]
                rows.append({'dataset':name,'split':split,'variant':'classical_signed_cosine','metric':'signed_cosine',
                  'k':k,'accuracy':float(np.mean(p==yte)),
                  'neighbour_jaccard_vs_euclidean':float(np.mean([jaccard(a,b) for a,b in zip(neigh,ref_neigh)])),
                  'prediction_disagreement_vs_euclidean':float(np.mean(p!=ref)),
                  'mean_rank_spearman_vs_euclidean':float(np.nanmean([spearman_safe(a,b) for a,b in zip(S_cos,S_euc)])),
                  'mean_boundary_gap':float(np.mean(np.sort(S_cos,axis=1)[:,::-1][:,k-1]-np.sort(S_cos,axis=1)[:,::-1][:,k])) if ntr>k else np.nan})

            variants,mu=build_variants(Xtr,Xte,args.anchor_strengths)
            for vname,(Atr,Ate,family,astrength) in variants.items():
                Utr,normtr=normalize_rows(Atr); Ute,normte=normalize_rows(Ate)
                signed=np.clip(Ute@Utr.T,-1,1)
                squared=np.clip(signed**2,0,1)  # exact real amplitude fidelity
                absolute=np.abs(signed)
                metric_mats={
                  'amplitude_fidelity_squared_cosine':squared,
                  'signed_cosine_diagnostic':signed,
                  'absolute_cosine_diagnostic':absolute,
                  'euclidean_on_unit_amplitudes':-np.linalg.norm(Ute[:,None,:]-Utr[None,:,:],axis=2),
                }
                for metric,S in metric_mats.items():
                    for k in args.k_values:
                        pred=knn_predict(S,ytr,k); neigh=[topk(r,k) for r in S]
                        jac=np.mean([jaccard(a,b) for a,b in zip(neigh,ref_neigh)])
                        corr=np.nanmean([spearman_safe(a,b) for a,b in zip(S,S_euc)])
                        sorted_s=np.sort(S,axis=1)[:,::-1]
                        gap=float(np.mean(sorted_s[:,k-1]-sorted_s[:,k])) if ntr>k else np.nan
                        rows.append({'dataset':name,'split':split,'variant':vname,'metric':metric,'k':k,
                          'accuracy':float(np.mean(pred==yte)),
                          'neighbour_jaccard_vs_euclidean':float(jac),
                          'prediction_disagreement_vs_euclidean':float(np.mean(pred!=ref)),
                          'mean_rank_spearman_vs_euclidean':float(corr),'mean_boundary_gap':gap})
                        if metric=='amplitude_fidelity_squared_cosine':
                            for qi in range(nte):
                                pairdiag.append({'dataset':name,'split':split,'variant':vname,'k':k,'query_index':qi,
                                  'negative_cosine_fraction':float(np.mean(signed[qi]<-1e-12)),
                                  'mean_signed_cosine':float(np.mean(signed[qi])),
                                  'fraction_topk_signed_cosine_negative':float(np.mean(signed[qi,topk(S[qi],k)]<0)),
                                  'topk_jaccard_vs_euclidean':float(jaccard(neigh[qi],ref_neigh[qi]))})
                # Norm-only and norm-aware controls, evaluated separately from
                # the amplitude-state metrics.
                norm_similarity=-np.abs(normte[:,None]-normtr[None,:])
                for k in args.k_values:
                    pn=knn_predict(norm_similarity,ytr,k); nn=[topk(r,k) for r in norm_similarity]
                    norm_rows.append({'dataset':name,'split':split,'variant':vname,'k':k,
                      'norm_only_accuracy':float(np.mean(pn==yte)),
                      'norm_only_neighbour_jaccard_vs_euclidean':float(np.mean([jaccard(a,b) for a,b in zip(nn,ref_neigh)])),
                      'train_norm_class_eta_squared':eta_squared(normtr,ytr),
                      'test_norm_class_eta_squared_descriptive':eta_squared(normte,yte),
                      'mean_train_norm':float(np.mean(normtr)),'sd_train_norm':float(np.std(normtr)),
                      'zero_train_vector_fraction':float(np.mean(normtr<EPS))})
                # State-level sign ambiguity diagnostics.
                pair_rows.append({'dataset':name,'split':split,'variant':vname,
                  'fraction_negative_signed_cosine_train_pairs':float(np.mean((Utr@Utr.T)[np.triu_indices(ntr,1)]<0)) if ntr>1 else np.nan,
                  'mean_signed_cosine_train_pairs':float(np.mean((Utr@Utr.T)[np.triu_indices(ntr,1)])) if ntr>1 else np.nan,
                  'amplitude_fidelity_squared_cosine_max_abs_error':float(np.max(np.abs(squared-signed**2))),
                  'train_norm_class_eta_squared':eta_squared(normtr,ytr),
                  'anchor_strength':astrength,'state_dimension':Atr.shape[1]})
                if family=='train_mean_plus_fixed_anchor':
                    for k in args.k_values:
                        ix=0
                        pred=knn_predict(squared,ytr,k)
                        anchor_rows.append({'dataset':name,'split':split,'anchor_strength':astrength,'k':k,
                          'accuracy':float(np.mean(pred==yte)),
                          'neighbour_jaccard_vs_euclidean':float(np.mean([jaccard(topk(r,k),b) for r,b in zip(squared,ref_neigh)])),
                          'negative_cosine_train_pair_fraction':pair_rows[-1]['fraction_negative_signed_cosine_train_pairs'],
                          'mean_pre_normalization_train_norm':float(np.mean(normtr))})
            # Class-label relationship of original scaled vector magnitude.
            raw_norm_tr=np.linalg.norm(Xtr,axis=1)
            try:
                groups=[raw_norm_tr[ytr==c] for c in np.unique(ytr)]
                pval=float(f_oneway(*groups).pvalue) if len(groups)>1 and all(len(g)>1 for g in groups) else np.nan
            except Exception: pval=np.nan
            dataset_rows.append({'dataset':name,'split':split,'n_train':ntr,'n_test':nte,'n_features_used':len(chosen),
              'selected_feature_indices':','.join(map(str,chosen.tolist())),
              'raw_scaled_norm_class_eta_squared':eta_squared(raw_norm_tr,ytr),
              'raw_scaled_norm_class_anova_p_value':pval,
              'preprocessing':'MinMaxScaler fit on training fold; test clipped to [0,1]; train-only variance feature selection'})
        print(f'  completed {args.n_splits} splits',flush=True)

    raw=pd.DataFrame(rows)
    summary=raw.groupby(['dataset','variant','metric','k'],dropna=False).agg(
      accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
      neighbour_jaccard_mean=('neighbour_jaccard_vs_euclidean','mean'),
      prediction_disagreement_mean=('prediction_disagreement_vs_euclidean','mean'),
      rank_spearman_mean=('mean_rank_spearman_vs_euclidean','mean'),
      boundary_gap_mean=('mean_boundary_gap','mean')).reset_index()
    summary['split_count']=args.n_splits
    pair=pd.DataFrame(pair_rows)
    pair_summary=pair.groupby(['dataset','variant'],dropna=False).mean(numeric_only=True).reset_index()
    norm=pd.DataFrame(norm_rows)
    norm_summary=norm.groupby(['dataset','variant','k'],dropna=False).mean(numeric_only=True).reset_index()
    anchor=pd.DataFrame(anchor_rows)
    if not anchor.empty:
        anchor_summary=anchor.groupby(['dataset','anchor_strength','k']).agg(
          accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
          neighbour_jaccard_mean=('neighbour_jaccard_vs_euclidean','mean'),
          negative_cosine_pair_fraction=('negative_cosine_train_pair_fraction','mean')).reset_index()
    else: anchor_summary=pd.DataFrame()
    config=pd.DataFrame([{'datasets':','.join(args.datasets),'n_splits':args.n_splits,'max_train':args.max_train,
      'max_test':args.max_test,'max_features':args.max_features,'k_values':','.join(map(str,args.k_values)),
      'anchor_strengths':','.join(map(str,args.anchor_strengths)),'seed':args.seed,
      'shot_simulation':False,'hardware':False,'interval_note':'SD across repeated, overlapping stratified splits; not independent-sample confidence intervals',
      'feature_selection':'Top variance features fitted on training fold only; no labels used',
      'amplitude_fidelity':'For real normalized vectors, F=|<xhat|yhat>|^2=cos(theta)^2',
      'anchor_map':'Normalize([x - training_mean, anchor_strength]); fixed appended coordinate; exploratory map, not claimed optimal'}])
    method=pd.DataFrame([
      ['Amplitude fidelity','Squared signed cosine after row L2 normalization','Exact ideal amplitude-state fidelity for real vectors'],
      ['Signed cosine','xhat dot yhat','Classical diagnostic; distinguishes opposite directions; not SWAP fidelity'],
      ['Absolute cosine','abs(xhat dot yhat)','Diagnostic for antipodal equivalence; not the same as squared cosine in all rankings'],
      ['Unit-vector Euclidean','-||xhat-yhat||','Classical diagnostic; monotonically related to signed cosine'],
      ['Fixed anchor','Normalize([x - train_mean, a])','Tests whether a reference coordinate offsets negative inner products'],
      ['Norm-only control','-abs(||x||-||y||)','Measures predictive/retrieval information in magnitude alone'],
      ['Leakage control','Scaler and feature ranking fit on train fold only','Test labels are not used for feature selection or preprocessing'],
      ['Scope','Exact geometry only','No shots, Aer, hardware noise, state-preparation cost, or QRAM model'],
    ],columns=['component','definition','interpretation'])
    return {'run_config':config,'methodology':method,'summary_by_dataset':summary,
      'results_by_split':raw,'sign_diagnostics':pair,'sign_diagnostics_summary':pair_summary,
      'norm_diagnostics_by_split':norm,'norm_diagnostics_summary':norm_summary,
      'anchor_results_by_split':anchor,'anchor_summary':anchor_summary,
      'query_sign_diagnostics':pd.DataFrame(pairdiag),'dataset_structure':pd.DataFrame(dataset_rows)}


def parse_args():
    p=argparse.ArgumentParser(description='Study 09: isolate amplitude-encoding geometry mechanisms')
    p.add_argument('--datasets',nargs='+',default=['iris','wine','breast_cancer','synthetic_directional','synthetic_magnitude'],
      choices=['iris','wine','breast_cancer','digits','synthetic_directional','synthetic_magnitude'])
    p.add_argument('--n-splits',type=int,default=20)
    p.add_argument('--test-size',type=float,default=.25)
    p.add_argument('--max-train',type=int,default=120)
    p.add_argument('--max-test',type=int,default=50)
    p.add_argument('--max-features',type=int,default=8)
    p.add_argument('--k-values',nargs='+',type=int,default=[1,3,5,9])
    p.add_argument('--anchor-strengths',nargs='+',type=float,default=[.25,.5,1.,2.])
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--outdir',default='results/study_09')
    p.add_argument('--workbook',default='study_09_amplitude_geometry.xlsx')
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if a.quick:
        a.datasets=['iris']; a.n_splits=2; a.max_train=45; a.max_test=20
        a.max_features=4; a.k_values=[1,3]; a.anchor_strengths=[.5,1.]
    if a.n_splits<2: p.error('--n-splits must be >= 2')
    if any(k<1 for k in a.k_values): p.error('k values must be positive')
    if any(x<=0 for x in a.anchor_strengths): p.error('anchor strengths must be positive')
    return a


def main():
    args=parse_args(); os.makedirs(args.outdir,exist_ok=True)
    print('='*76); print('STUDY 09 — AMPLITUDE-ENCODING GEOMETRY MECHANISMS'); print('='*76)
    sheets=run(args); path=os.path.join(args.outdir,args.workbook)
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,df in sheets.items():
            if df is None: continue
            sn=name[:31]; df.to_excel(writer,sheet_name=sn,index=False)
            ws=writer.sheets[sn]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                vals=[len(str(c.value)) if c.value is not None else 0 for c in col[:200]]
                ws.column_dimensions[col[0].column_letter].width=min(max(12,max(vals,default=10)+2),48)
    print('\nCOMPLETE'); print(f'Workbook: {os.path.abspath(path)}')
    print('Sheets: '+', '.join(sheets.keys()))

if __name__=='__main__':
    main()
