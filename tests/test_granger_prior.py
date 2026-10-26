import numpy as np
import pytest


def test_conditional_var_direction():
    from scripts.build_granger_prior import estimate_window
    rng=np.random.default_rng(55)
    x=np.zeros((2000,3))
    for t in range(1,len(x)):
        x[t,0]=.5*x[t-1,0]+rng.normal()
        x[t,1]=.8*x[t-1,0]+.2*x[t-1,1]+rng.normal(scale=.3)
        x[t,2]=rng.normal()
    score,p=estimate_window(x,2)
    assert score[1,0]>.5 and p[1,0]<1e-6
    assert score[0,1]<.02 and p[0,1]>.01
    assert np.isfinite(score).all()


def test_constant_variables_safe():
    from scripts.build_granger_prior import estimate_window
    x=np.ones((200,3)); x[:,1]=np.random.default_rng(3).normal(size=200)
    score,p=estimate_window(x,2)
    assert np.isfinite(score).all() and np.isfinite(p).all()
    assert np.all(score[0]==0)


def test_exact_lagged_prediction_is_not_dropped():
    from scripts.build_granger_prior import estimate_window
    rng=np.random.default_rng(123)
    x=rng.normal(size=(1000,2)); x[1:,1]=x[:-1,0]
    score,p=estimate_window(x,1)
    assert score[1,0]>.99 and p[1,0]<1e-10


def test_regularization_zoh_preserves_steps_and_segments():
    import pandas as pd
    from scripts.regularize_alfa import regularize_split
    df=pd.DataFrame(dict(time_sec=[0,.04,.11,0,.1],wp_dist=[1,2,8,10,20],label=[0,0,1,0,0]))
    meta=pd.DataFrame(dict(segment_id=[0,0,0,1,1],original_index=range(5),time_sec=df.time_sec))
    out,m=regularize_split(df,meta,20,.25)
    assert set(out.wp_dist).issubset(set(df.wp_dist))
    assert out[m.segment_id==1].wp_dist.iloc[0]==10
    assert (m.original_time_sec<=m.time_sec+1e-9).all()
    assert set(m.original_index).issubset(set(meta.original_index))


def test_events_keep_original_onset_and_segment_boundaries():
    import pandas as pd
    from utils.experiment_reporting import event_details
    labels=np.array([0,1,1,1,1,1,0,0])
    meta=pd.DataFrame(dict(segment_id=[0,0,0,0,1,1,1,1],time_sec=[0,.05,.1,.15,0,.05,.1,.15]))
    rows=event_details(labels,np.array([1,0,0,1]),np.array([2,3,4,5]),meta)
    assert len(rows)==2
    assert rows[0]['delay_points']==1 and rows[0]['delay_seconds']==pytest.approx(.05)
    assert rows[1]['delay_points']==1 and rows[1]['delay_seconds']==pytest.approx(.05)


def test_prior_reader_uses_normal_training_flights_only(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from scripts.preprocess_alfa import FEATURE_COLUMNS
    from scripts.build_granger_prior import training_segments
    n=300
    df=pd.DataFrame(np.random.default_rng(2).normal(size=(n,10)),columns=FEATURE_COLUMNS)
    df.insert(0,'time_sec',np.tile(np.arange(150)/20,2)); df['label']=0
    df.to_csv(tmp_path/'train.csv',index=False)
    pd.DataFrame(dict(segment_id=[0]*150+[1]*150,time_sec=df.time_sec,label=df.label,
        source_file=['normal.csv']*150+['fault.csv']*150,fault_type=['normal']*150+['engine']*150,
        original_index=np.arange(n))).to_csv(tmp_path/'train_meta.csv',index=False)
    (tmp_path/'metadata.json').write_text(json.dumps(dict(split_policy='fine_grained',sampling_mode='regular',sampling_hz=20)))
    (tmp_path/'val.csv').touch()  # existence only; contents must never be opened
    original=pd.read_csv
    def read_only_training(path,*a,**kw):
        assert path.name in {'train.csv','train_meta.csv'}
        return original(path,*a,**kw)
    monkeypatch.setattr(pd,'read_csv',read_only_training)
    segments=list(training_segments(tmp_path))
    assert len(segments)==1 and len(segments[0][1])==150
    assert segments[0][2].source_file.eq('normal.csv').all()
