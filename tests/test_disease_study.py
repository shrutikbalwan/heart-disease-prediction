"""Disease-study integrity, threshold, calibration and subgroup checks."""
import hashlib,json
import numpy as np
import pandas as pd
from functools import partial
read_csv=partial(pd.read_csv,float_precision="round_trip")
import pytest
from ml_projects.disease_study import ROOT,OUT,PROTOCOL,bp_invalid,wilson,choose_threshold,counts,probability_metrics,calibration_bins,decision
from ml_projects.data import read_dataset,split_target,PrepareFeatures

@pytest.fixture(scope='module')
def source():return split_target(read_dataset('disease'),'disease')

@pytest.fixture(scope='module')
def predictions():return read_csv(OUT/'out_of_fold_predictions.csv')

def test_bp_rule_and_cleaning_are_row_local(source):
    X,_=source;sample=X.head(6).copy();original=sample.copy(deep=True)
    sample['ap_hi']=[120,70,0,301,120,120];sample['ap_lo']=[80,120,80,80,201,120]
    np.testing.assert_array_equal(bp_invalid(sample),[False,True,True,True,True,True])
    clean=PrepareFeatures('disease',clean_bp=True).fit_transform(sample)
    raw=PrepareFeatures('disease',clean_bp=False).fit_transform(sample)
    assert clean.iloc[1:][['ap_hi','ap_lo']].isna().all().all()
    assert clean.iloc[0].ap_hi==120 and clean.iloc[0].ap_lo==80
    pd.testing.assert_frame_equal(clean.drop(columns=['ap_hi','ap_lo']),raw.drop(columns=['ap_hi','ap_lo']))
    pd.testing.assert_series_equal(sample.ap_hi,pd.Series([120,70,0,301,120,120],index=sample.index,name='ap_hi'))
    pd.testing.assert_frame_equal(original,X.head(6))

def test_uncertainty_and_threshold_edge_cases():
    assert wilson(0,0)==(None,None)
    low,high=wilson(0,10);assert low==pytest.approx(0);assert high==pytest.approx(.2775328,abs=1e-6)
    low,high=wilson(10,10);assert low==pytest.approx(.7224672,abs=1e-6);assert high==pytest.approx(1)
    y=np.array([0,0,1,1,1]);p=np.array([.1,.4,.3,.7,.9])
    thresholds=[choose_threshold(y,p,target) for target in [.7,.8,.9]]
    assert thresholds==sorted(thresholds,reverse=True)
    for target,t in zip([.7,.8,.9],thresholds):assert counts(y,p,t)['sensitivity']>=target
    with pytest.raises(ValueError):choose_threshold(np.array([0,0]),np.array([.1,.2]),.8)

def test_partitions_and_out_of_fold_coverage(predictions):
    original=json.loads((ROOT/'outputs/split.json').read_text());train=set(original['train_rows']);test=set(original['test_rows']);evaluation=[]
    parts=json.loads((OUT/'partitions.json').read_text())
    for part in parts:
        sets=[set(part[k]) for k in ['fit','calibration','threshold_validation','evaluation']]
        assert set.union(*sets)==train
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        assert all(not a&test for a in sets);evaluation.extend(part['evaluation'])
    assert len(evaluation)==len(set(evaluation))==len(train)
    assert set(predictions.variant)=={'logistic_clean','logistic_raw','boosting_clean','boosting_raw'}
    for _,frame in predictions.groupby('variant'):
        assert set(frame.source_row)==train and not frame.source_row.duplicated().any()
        assert frame[['raw_probability','calibrated_probability']].ge(0).all().all()
        assert frame[['raw_probability','calibrated_probability']].le(1).all().all()

def test_probability_and_classification_metrics(source,predictions):
    _,y=source;folds=read_csv(OUT/'outer_fold_metrics.csv')
    for (fold,variant),frame in predictions.groupby(['fold','variant']):
        np.testing.assert_array_equal(frame.actual,y.loc[frame.source_row].to_numpy())
        row=folds[(folds.fold==fold)&(folds.variant==variant)].iloc[0]
        for stage in ['raw','calibrated']:
            for metric,value in probability_metrics(frame.actual,frame[stage+'_probability']).items():assert row[stage+'_'+metric]==pytest.approx(value)
        for metric,value in counts(frame.actual,frame.calibrated_probability,row.primary_threshold).items():assert row[metric]==pytest.approx(value)
        np.testing.assert_array_equal(frame.prediction,(frame.calibrated_probability>=row.primary_threshold).astype(int))

def test_thresholds_chosen_on_validation_only(predictions):
    validation=read_csv(OUT/'threshold_validation_predictions.csv');policies=read_csv(OUT/'threshold_policies.csv')
    parts={p['fold']:p for p in json.loads((OUT/'partitions.json').read_text())}
    grid=np.array(PROTOCOL['threshold_grid'])
    for (fold,variant),frame in validation.groupby(['fold','variant']):
        assert set(frame.source_row)==set(parts[fold]['threshold_validation'])
        outer=predictions[(predictions.fold==fold)&(predictions.variant==variant)]
        positive=frame[frame.actual==1].probability.to_numpy()
        sensitivity=(positive[:,None]>=grid).mean(axis=0)
        for target in PROTOCOL['sensitivity_targets']:
            expected=float(grid[sensitivity>=target].max())
            row=policies[(policies.fold==fold)&(policies.variant==variant)&(policies.policy==f'validation_sensitivity_{target:.0%}')].iloc[0]
            assert row.threshold==pytest.approx(expected)
            for metric,value in counts(outer.actual,outer.calibrated_probability,expected).items():assert row[metric]==pytest.approx(value)

def test_calibration_bins_recomputed(predictions):
    bins=read_csv(OUT/'calibration_bins.csv')
    for variant,frame in predictions.groupby('variant'):
        for stage in ['raw','calibrated']:
            actual=bins[(bins.variant==variant)&(bins.stage==stage)].sort_values('bin')
            expected=pd.DataFrame(calibration_bins(frame.actual.to_numpy(),frame[stage+'_probability'].to_numpy())).sort_values('bin')
            assert int(actual.rows.sum())==len(frame)
            np.testing.assert_allclose(actual[expected.columns].to_numpy(),expected.to_numpy(),rtol=1e-10,atol=1e-10)

def test_subgroup_counts_intervals_and_source_codes(source,predictions):
    X,_=source;groups=read_csv(OUT/'subgroups.csv',dtype={'group':str})
    for (variant,attribute),frame in groups.groupby(['variant','attribute']):assert frame.rows.sum()==56000
    for _,row in groups.iterrows():
        assert row.positives+row.negatives==row.rows
        assert row.TP+row.FN==row.positives and row.TN+row.FP==row.negatives
        assert row.small_sample==(row.rows<100 or min(row.positives,row.negatives)<30)
        if row.positives:
            lo,hi=wilson(row.TP,row.positives);assert row.sensitivity_low==pytest.approx(lo);assert row.sensitivity_high==pytest.approx(hi)
        if row.negatives:
            lo,hi=wilson(row.TN,row.negatives);assert row.specificity_low==pytest.approx(lo);assert row.specificity_high==pytest.approx(hi)
    for variant,frame in predictions.groupby('variant'):
        codes=X.loc[frame.source_row,'gender'].astype(str).to_numpy()
        for code in np.unique(codes):
            part=frame[codes==code];row=groups[(groups.variant==variant)&(groups.attribute=='gender_code')&(groups['group']==code)].iloc[0]
            assert row.rows==len(part)
            assert row.TP==int(((part.actual==1)&(part.prediction==1)).sum())

def test_model_decision_and_active_artifacts():
    summary=json.loads((OUT/'summary.json').read_text());folds=read_csv(OUT/'outer_fold_metrics.csv');result=decision(folds)
    for key,value in result.items():
        if isinstance(value,float):assert summary['decision'][key]==pytest.approx(value)
        else:assert summary['decision'][key]==value
    for name,digest in summary['active_model_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert 'No clinical' in (ROOT/'DISEASE_VALIDATION_REPORT.html').read_text(encoding='utf-8')


def test_offline_calibration_chart():
    chart=(OUT/'calibration_tradeoffs.html').read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in chart and 'boosting_clean' in chart
    assert '<script src=' not in chart
