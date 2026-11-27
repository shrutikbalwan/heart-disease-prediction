"""Nested disease validation and blood-pressure sensitivity study.
Educational evaluation only. Run: python -m ml_projects.disease_study
"""
import hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
from html import escape
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
from .data import ROOT,FILES,read_dataset,split_target
from .train import pipeline,write_json

OUT=ROOT/'outputs'/'disease_study'
PROTOCOL={'seed':20261127,'outer_folds':5,'inner_folds':3,'fit_fraction':.6,'calibration_fraction':.2,'threshold_validation_fraction':.2,'selection_metric':'neg_log_loss','logistic_C':[.1,1.],'boosting_leaf_nodes':[15,31],'boosting_iterations':150,'sensitivity_targets':[.7,.8,.9],'primary_target':.8,'threshold_grid':[round(float(t),2) for t in np.linspace(0,1,101)],'calibration_method':'sigmoid','age_bands':['under 40','40-54','55-69','70+'],'small_group_rows':100,'small_class_rows':30,'scope':'Original training rows only; all original historical-test rows excluded. Previously studied data, internal validation only. No clinical, diagnostic, causal or future-onset claims.','family_support_rule':'Cleaned boosting must have lower mean log loss than cleaned logistic and win at least 4/5 folds. This is an engineering comparison, not a significance test.','raw_bp_review_rule':'Raw-BP boosting requires mean log-loss improvement >=0.005, AUC improvement >=0.005 and lower log loss in at least 4/5 folds to trigger further review. No automatic policy change.'}

def bp_invalid(X):
    hi=pd.to_numeric(X.ap_hi,errors='coerce');lo=pd.to_numeric(X.ap_lo,errors='coerce')
    return (hi<=0)|(lo<=0)|(hi>300)|(lo>200)|(lo>=hi)

def wilson(successes,total):
    if total==0:return (None,None)
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return float(max(0,center-half)),float(min(1,center+half))

def counts(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,np.asarray(p)>=threshold,labels=[0,1]).ravel()
    return {'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,'precision':float(tp/(tp+fp)) if tp+fp else 0.,'F1':float(f1_score(y,np.asarray(p)>=threshold,zero_division=0))}

def probability_metrics(y,p):
    return {'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'log_loss':float(log_loss(y,p,labels=[0,1])),'Brier':float(brier_score_loss(y,p))}

def choose_threshold(y,p,target):
    candidates=[{'threshold':t,**counts(y,p,t)} for t in PROTOCOL['threshold_grid']]
    eligible=[r for r in candidates if r['sensitivity'] is not None and r['sensitivity']>=target]
    if not eligible:raise ValueError('Sensitivity selection requires positive validation outcomes')
    return float(max(eligible,key=lambda r:(r['specificity'] if r['specificity'] is not None else -1,r['threshold']))['threshold'])

def calibration_bins(y,p):
    frame=pd.DataFrame({'actual':np.asarray(y),'probability':p});frame['bin']=np.minimum((np.asarray(p)*10).astype(int),9)
    rows=[]
    for index,data in frame.groupby('bin'):
        positives=int(data.actual.sum());lo,hi=wilson(positives,len(data))
        rows.append({'bin':int(index),'rows':len(data),'mean_probability':float(data.probability.mean()),'observed_frequency':float(data.actual.mean()),'frequency_low':lo,'frequency_high':hi})
    return rows

def subgroup_table(predictions,X):
    rows=[]
    for variant,frame in predictions.groupby('variant'):
        ids=frame.source_row.to_numpy(); source=X.loc[ids]
        age=pd.cut(source.age/365.25,[-np.inf,40,55,70,np.inf],labels=PROTOCOL['age_bands'],right=False).astype(str).to_numpy()
        code=source.gender.astype(str).to_numpy();status=np.where(bp_invalid(source),'invalid_imputed_if_cleaned','within_rule')
        groups={'age_band':age,'gender_code':code,'bp_status':status,'age_x_gender':np.char.add(np.char.add(age.astype(str),' / '),code.astype(str))}
        for attribute,values in groups.items():
            for value in np.unique(values):
                group=frame[np.asarray(values)==value]; y=group.actual.to_numpy();p=group.calibrated_probability.to_numpy()
                # Each row uses the threshold selected by its own outer fold.
                stats=counts(y,group.prediction.to_numpy(),.5);pos=stats['TP']+stats['FN'];neg=stats['TN']+stats['FP']
                senslo,senshi=wilson(stats['TP'],pos);speclo,spechi=wilson(stats['TN'],neg)
                rows.append({'variant':variant,'attribute':attribute,'group':str(value),'rows':len(group),'positives':pos,'negatives':neg,'small_sample':bool(len(group)<100 or min(pos,neg)<30),**stats,**probability_metrics(y,p),'sensitivity_low':senslo,'sensitivity_high':senshi,'specificity_low':speclo,'specificity_high':spechi})
    return pd.DataFrame(rows)

def decision(folds):
    loss=folds.pivot(index='fold',columns='variant',values='calibrated_log_loss');auc=folds.pivot(index='fold',columns='variant',values='calibrated_ROC_AUC')
    advantage=loss.logistic_clean-loss.boosting_clean;raw_gain=loss.boosting_clean-loss.boosting_raw
    raw_auc=auc.boosting_raw-auc.boosting_clean
    return {'cleaned_boosting_family_supported':bool(advantage.mean()>0 and (advantage>0).sum()>=4),'cleaned_boosting_fold_wins':int((advantage>0).sum()),'mean_log_loss_advantage_over_logistic':float(advantage.mean()),'raw_bp_requires_further_review':bool(raw_gain.mean()>=.005 and raw_auc.mean()>=.005 and (raw_gain>0).sum()>=4),'raw_bp_mean_log_loss_gain':float(raw_gain.mean()),'raw_bp_mean_AUC_gain':float(raw_auc.mean()),'active_model_action':'Retain current model and cleaning policy; no automated deployment from this study. Any proposed change needs a separately verified final fit and domain/data review.'}

def run():
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'protocol.json',PROTOCOL)
    tracked=[ROOT/'outputs/model.joblib',ROOT/'outputs/advanced/model.joblib',ROOT/'outputs/advanced/summary.json'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    X,y=split_target(read_dataset('disease'),'disease');X=X.drop(columns=['id','date','country','occupation'])
    original=json.loads((ROOT/'outputs/split.json').read_text());ids=np.array(original['train_rows']);excluded=set(original['test_rows']);assert not set(ids)&excluded
    partitions=[];folds=[];predictions=[];validation=[];policies=[];selections=[];start=time.time()
    outer=StratifiedKFold(5,shuffle=True,random_state=PROTOCOL['seed'])
    for fold,(train_pos,eval_pos) in enumerate(outer.split(ids,y.loc[ids]),1):
        train,evaluation=ids[train_pos],ids[eval_pos];seed=PROTOCOL['seed']+fold
        fit,held=train_test_split(train,test_size=.4,stratify=y.loc[train],random_state=seed)
        cal,val=train_test_split(held,test_size=.5,stratify=y.loc[held],random_state=seed)
        partitions.append({'fold':fold,'fit':fit.tolist(),'calibration':cal.tolist(),'threshold_validation':val.tolist(),'evaluation':evaluation.tolist()})
        for family in ['logistic','boosting']:
            for clean in [True,False]:
                variant=family+('_clean' if clean else '_raw');print(f'Outer fold {fold}/5: {variant}',flush=True)
                estimator=LogisticRegression(max_iter=3000,random_state=seed) if family=='logistic' else HistGradientBoostingClassifier(max_iter=150,random_state=seed)
                grid={'model__C':PROTOCOL['logistic_C']} if family=='logistic' else {'model__max_leaf_nodes':PROTOCOL['boosting_leaf_nodes'],'model__l2_regularization':[1.]}
                inner=StratifiedKFold(3,shuffle=True,random_state=seed)
                search=GridSearchCV(pipeline('disease',X.loc[fit],estimator,clean_bp=clean),grid,cv=inner,scoring='neg_log_loss',n_jobs=1,error_score='raise')
                search.fit(X.loc[fit],y.loc[fit]);base=search.best_estimator_
                pd.DataFrame(search.cv_results_).to_csv(OUT/f'fold_{fold}_{variant}_inner.csv',index=False)
                selections.append({'fold':fold,'variant':variant,'parameters':search.best_params_,'inner_cv_score':float(search.best_score_)})
                calibrated=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid',n_jobs=1);calibrated.fit(X.loc[cal],y.loc[cal])
                vp=calibrated.predict_proba(X.loc[val])[:,1];p=calibrated.predict_proba(X.loc[evaluation])[:,1];raw=base.predict_proba(X.loc[evaluation])[:,1];truth=y.loc[evaluation].to_numpy()
                thresholds={target:choose_threshold(y.loc[val].to_numpy(),vp,target) for target in PROTOCOL['sensitivity_targets']};threshold=thresholds[.8]
                for row,actual,prob in zip(val,y.loc[val],vp):validation.append({'fold':fold,'variant':variant,'source_row':int(row),'actual':int(actual),'probability':float(prob)})
                folds.append({'fold':fold,'variant':variant,'primary_threshold':threshold,**{f'raw_{k}':v for k,v in probability_metrics(truth,raw).items()},**{f'calibrated_{k}':v for k,v in probability_metrics(truth,p).items()},**counts(truth,p,threshold)})
                for label,t in [('fixed_0.5',.5),*[(f'validation_sensitivity_{target:.0%}',thresholds[target]) for target in thresholds]]:
                    policies.append({'fold':fold,'variant':variant,'policy':label,'threshold':t,**counts(truth,p,t)})
                for row,actual,before,after in zip(evaluation,truth,raw,p):predictions.append({'fold':fold,'variant':variant,'source_row':int(row),'actual':int(actual),'raw_probability':float(before),'calibrated_probability':float(after),'threshold':threshold,'prediction':int(after>=threshold)})
    fold_df=pd.DataFrame(folds);pred_df=pd.DataFrame(predictions);subgroups=subgroup_table(pred_df,X)
    curves=[]
    for variant,frame in pred_df.groupby('variant'):
        for stage in ['raw','calibrated']:
            curves.extend({'variant':variant,'stage':stage,**r} for r in calibration_bins(frame.actual.to_numpy(),frame[stage+'_probability'].to_numpy()))
    for name,frame in [('outer_fold_metrics.csv',fold_df),('out_of_fold_predictions.csv',pred_df),('threshold_validation_predictions.csv',pd.DataFrame(validation)),('threshold_policies.csv',pd.DataFrame(policies)),('subgroups.csv',subgroups),('calibration_bins.csv',pd.DataFrame(curves))]:frame.to_csv(OUT/name,index=False)
    write_json(OUT/'partitions.json',partitions);write_json(OUT/'inner_selections.json',selections)
    numeric=[c for c in fold_df.select_dtypes(include='number') if c!='fold'];means=fold_df.groupby('variant')[numeric].mean().reset_index().to_dict('records')
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,'original_training_rows':len(ids),'excluded_historical_test_rows':len(excluded),'invalid_bp_training_rows':int(bp_invalid(X.loc[ids]).sum()),'missing_bp_training_rows':int(X.loc[ids,['ap_hi','ap_lo']].isna().any(axis=1).sum()),'mean_outer_metrics':means,'decision':decision(fold_df),'active_models_unchanged':True,'active_model_hashes':hashes,'dataset_sha256':hashlib.sha256((ROOT/'datasets'/FILES['disease']).read_bytes()).hexdigest(),'study_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start}
    write_json(OUT/'summary.json',summary);write_report(summary,subgroups,pd.DataFrame(curves),pd.DataFrame(policies))
    print(json.dumps({'means':means,'decision':summary['decision']},indent=2),flush=True)
    return summary

def write_report(summary,subgroups,curves,policies):
    table=lambda rows:pd.DataFrame(rows).to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure=make_subplots(rows=1,cols=2,subplot_titles=['Calibration on outer evaluation rows','Sensitivity-specificity trade-offs'],horizontal_spacing=.14)
    colors={'boosting_clean':'#146b62','boosting_raw':'#d97706','logistic_clean':'#286ab4','logistic_raw':'#a23c85'}
    figure.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line={'color':'#8a98a4','dash':'dot'},name='Ideal calibration'),row=1,col=1)
    for variant,color in colors.items():
        for stage in ['raw','calibrated']:
            data=curves[(curves.variant==variant)&(curves.stage==stage)]
            figure.add_trace(go.Scatter(x=data.mean_probability,y=data.observed_frequency,mode='lines+markers',name=variant+' / '+stage,line={'color':color,'dash':'dash' if stage=='raw' else 'solid'},customdata=data.rows,hovertemplate='Mean score %{x:.3f}<br>Observed frequency %{y:.3f}<br>Rows %{customdata}<extra>%{fullData.name}</extra>'),row=1,col=1)
        trade=policies[policies.variant==variant].groupby('policy')[['sensitivity','specificity']].mean().reset_index().sort_values('sensitivity')
        figure.add_trace(go.Scatter(x=trade.sensitivity,y=trade.specificity,mode='lines+markers',name=variant+' / policies',line={'color':color},customdata=trade.policy,hovertemplate='%{customdata}<br>Sensitivity %{x:.1%}<br>Specificity %{y:.1%}<extra>%{fullData.name}</extra>'),row=1,col=2)
    figure.update_layout(title={'text':'Disease validation: calibration and decision trade-offs','x':.04},width=1250,height=760,template='plotly_white',legend={'orientation':'h','y':-.22,'x':0},margin={'t':100,'b':220})
    figure.update_xaxes(title_text='Mean predicted probability',range=[0,1],row=1,col=1);figure.update_yaxes(title_text='Observed frequency',range=[0,1],row=1,col=1)
    figure.update_xaxes(title_text='Sensitivity',range=[.5,1],row=1,col=2);figure.update_yaxes(title_text='Specificity',range=[.2,.9],row=1,col=2)
    figure.write_html(OUT/'calibration_tradeoffs.html',include_plotlyjs=True,config={'responsive':False,'displaylogo':False})
    columns=['variant','raw_log_loss','calibrated_log_loss','raw_Brier','calibrated_Brier','calibrated_ROC_AUC','sensitivity','specificity','F1']
    body='<h1>Disease model validation study</h1><p>Educational evaluation only. No clinical diagnosis, future-risk prediction or treatment recommendation. Original historical-test rows were excluded; this remains internal validation of previously studied data.</p><h2>Mean outer-fold results</h2>'+table(pd.DataFrame(summary['mean_outer_metrics'])[columns])+'<h2>Recorded decision</h2><pre>'+escape(json.dumps(summary['decision'],indent=2))+'</pre><h2>Sensitivity and specificity trade-offs</h2><p>Thresholds were chosen on validation rows and then evaluated on held-out outer folds. A target such as 80% validation sensitivity does not guarantee the same sensitivity on other data. Fixed 0.5 is shown as a reference.</p>'+table(policies.groupby(['variant','policy'])[['sensitivity','specificity','precision','F1','FP','FN']].mean().reset_index())+'<h2>Calibration evidence</h2><p>Raw and sigmoid-calibrated probabilities are evaluated on the same outer rows. Brier score and log loss are proper probability scores; lower is better. Calibration is not assumed to improve them. The following reliability bins use fixed probability deciles and contain counts and approximate Wilson intervals for observed event frequency.</p>'+table(curves)+'<h2>Age and dataset gender-code evaluation</h2><p>Gender codes retain their original values; no meanings are invented. Age is converted from days using 365.25. No age-70-plus training rows were available in this study; performance for that band is unmeasured. Subgroups use primary 80%-target fold thresholds. Wilson 95% intervals describe uncertainty in sensitivity/specificity under an independent-binomial approximation; they do not capture model-selection uncertainty or correlated observations. A group is flagged when it has fewer than 100 records or fewer than 30 positives or negatives.</p>'+table(subgroups[subgroups.attribute.isin(['age_band','gender_code'])])+'<h2>Blood-pressure data-quality sensitivity</h2><p>'+str(summary['invalid_bp_training_rows'])+' training rows violate the existing broad BP rule. The clean variants replace both BP fields with missing values and impute using fitting-row medians. Raw variants retain finite numeric anomalies as a sensitivity experiment; this is not a recommendation to treat invalid readings as valid. Missing inputs are imputed in either variant. No records are dropped.</p>'+table(subgroups[subgroups.attribute=='bp_status'])+'<h2>Age/code intersections</h2>'+table(subgroups[subgroups.attribute=='age_x_gender'])+'<h2>Design and limitations</h2><p>Five stratified outer folds, three inner search folds. Each outer development partition is split 60% fitting, 20% calibration and 20% threshold validation, identically across all four variants. Learned imputation, scaling and encoding stay inside each search fold. Inner selection uses log loss. FrozenEstimator prevents the calibration step from refitting the base classifier. Model-family and cleaning-policy comparisons are descriptive and do not imply clinical validity.</p><p>The existing BP bounds are data-quality assumptions, not diagnostic criteria. Height units remain unresolved; BMI is omitted. ID, date, country and occupation are excluded. Cross-validation folds share training observations; subgroup differences are not proof of unfairness or causal effects. No external dataset or temporal cohort was acquired. Current application models and policies remain unchanged.</p><p>Reproduce: <code>python -m ml_projects.disease_study</code>. Exact partitions, inner searches, probabilities, calibration bins and thresholds are in outputs/disease_study/. <a href="https://scikit-learn.org/stable/modules/calibration">Calibration method reference</a>.</p>'
    style='body{font:16px/1.6 Segoe UI,sans-serif;color:#163044;max-width:1380px;margin:40px auto;padding:0 24px;background:#f5f8fa}h1{font-size:36px}h2{margin-top:34px}table{display:block;overflow:auto;border-collapse:collapse;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d9e4ea;text-align:left}th{background:#e5eff3}pre{padding:20px;background:#e5eff3;overflow:auto}a{color:#146b62}'
    (ROOT/'DISEASE_VALIDATION_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Disease validation study</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
