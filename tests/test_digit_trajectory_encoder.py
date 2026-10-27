"""Small CPU tests for causal windows, private-data isolation and frozen keys."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import torch

from capx.memory.trajectory import data as d
from capx.memory.trajectory.model import FrozenTrajectoryEncoder, TrajectoryEncoder, augment, search_index, unit_rows, vicreg
from capx.memory.trajectory import train as tr


def episode(root,seed,n=32):
    path=root/f"{seed}_expert"/"public.h5"
    path.parent.mkdir(parents=True)
    reference=np.full((240,320),.03,np.float32)
    depths=np.broadcast_to(reference,(n,240,320)).copy()
    for i in range(n):
        depths[i,80:160,100:220]-=(1+np.sin(i*.25))*.00015
    motion=np.zeros((n,4),np.float32)
    motion[:,0]=seed*.000001
    motion[:,3]=np.arange(n)*.002
    with h5py.File(path,"w") as f:
        f.attrs['metadata']=json.dumps(dict(seed=seed,piano_sha256='test_sensor',physics_dt_s=1/240,secret_native_score=123))
        f['time_s']=np.arange(n)/30
        f['physics_step']=np.arange(n)*8
        f['action_target']=motion
        f['hand_state']=np.c_[motion,np.zeros((n,17),np.float32)]
        f['tactile/index/reference_depth_m']=reference
        f['tactile/index/depth_m']=depths
        f['tactile/index/valid_mask']=np.ones_like(depths,np.uint8)
        f['requested_midi']=np.full(n,60,np.int16)
    return path


@pytest.fixture
def prepared(tmp_path):
    root=tmp_path/'inputs'
    for seed in (3100,3130,3140,3200):episode(root,seed)
    cache=tmp_path/'run'/'cache'
    d.prepare(cache,[root],require_full=False)
    return d.Corpus(cache)


def test_causality_gaps_and_invalid_frames():
    times=np.arange(32)/30
    w=d.causal_windows(times,np.ones(32,bool))
    assert w.shape==(17,16)
    targets=times[w[:,-1],None]-np.arange(15,-1,-1)[None]/30
    assert np.all(times[w]<=targets+1e-10)
    future=d.causal_windows(np.arange(40)/30,np.ones(40,bool))
    np.testing.assert_array_equal(w,future[:len(w)])
    good=np.ones(32,bool);good[20]=False
    assert not np.any(d.causal_windows(times,good)==20)
    gapped=np.delete(times,np.arange(10,15))
    for ids in d.causal_windows(gapped,np.ones(len(gapped),bool)):
        assert np.diff(gapped[ids]).max()<=1/30+1e-10


def test_weighted_signed_depth_and_quality():
    ref=np.full((240,320),.03,np.float32)
    depth=np.stack([ref-.001,ref+.001])
    mask=np.ones_like(depth,np.uint8);mask[:,:,:160]=0
    maps,features,good=d.reduce_depth(depth,ref,mask)
    assert not good.any()
    assert maps[0,0,:,20:].mean()==pytest.approx(1,abs=1e-4)
    assert maps[1,0,:,20:].mean()==pytest.approx(-1,abs=1e-4)
    assert np.all(maps[:,:, :, :16]==0)
    assert features[:,0].mean()==pytest.approx(1,abs=1e-4)
    depth[0]=np.nan
    maps,_,good=d.reduce_depth(depth,ref,mask)
    assert not good[0] and np.isfinite(maps).all()


def test_public_only_preparation_and_training(tmp_path,monkeypatch):
    root=tmp_path/'inputs'
    for seed in (3100,3130,3140,3200):episode(root,seed)
    original=h5py.File
    opened=[]
    def guarded(path,*args,**kwargs):
        assert Path(path).name=='public.h5',f'Private file opened: {path}'
        opened.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(h5py,'File',guarded)
    run=tmp_path/'run'
    d.prepare(run/'cache',[root],require_full=False)
    corpus=d.Corpus(run/'cache')
    assert {e['seed']:e['split'] for e in corpus.episodes}=={3100:'train',3130:'dev',3140:'test',3200:'external'}
    assert corpus.norm['fitted_seeds']==[3100]
    assert 'secret_native_score' not in json.dumps(corpus.manifest)
    monkeypatch.setitem(tr.CONFIG,'max_epochs',1)
    monkeypatch.setitem(tr.CONFIG,'training_seeds',[0])
    torch.set_num_threads(1)
    summaries=tr.train_suite(run/'cache',run,'cpu')
    assert len(summaries)==2 and all(s['epochs']==1 for s in summaries)
    assert all(s['parameter_count']<100000 for s in summaries)
    from capx.memory.trajectory.evaluate import evaluate_suite
    results=evaluate_suite(run)
    assert len(results)==4
    assert (run/'evaluation/summary.json').exists()
    assert not (run/'private_audit').exists()
    assert opened
    # Native files do not even exist until both models and public evaluation finish.
    monkeypatch.setattr(h5py,'File',original)
    frozen_before=d.read_json(run/'freeze.json')
    for ep in corpus.episodes:
        with original(ep['source'],'r') as f: last=int(f['physics_step'][-1])
        with original(Path(ep['source']).with_name('private_labels.h5'),'w') as f:
            f['physics_step']=np.arange(last+1)
            forces=np.zeros((last+1,88),np.float32)
            forces[:,39]=np.linspace(0,.8,last+1)
            f['key_contact_force_N']=forces
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/opentac/audit_digit_trajectory_neighbors.py'
    spec=importlib.util.spec_from_file_location('private_trajectory_audit_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.audit(run)
    assert d.read_json(run/'freeze.json')==frozen_before
    assert (run/'REPORT.md').exists()


def test_normalization_is_train_only(tmp_path):
    root=tmp_path/'inputs'
    episode(root,3100);dev=episode(root,3130)
    d.prepare(tmp_path/'a',[root],require_full=False)
    with h5py.File(dev,'r+') as f:
        f['tactile/index/depth_m'][:]=.01
        f['action_target'][:]=100
    d.prepare(tmp_path/'b',[root],require_full=False)
    assert d.read_json(tmp_path/'a/normalization.json')==d.read_json(tmp_path/'b/normalization.json')


def test_duplicate_episode_cross_split_rejected(tmp_path):
    root=tmp_path/'inputs';source=episode(root,3100)
    target=root/'3130_expert'/'public.h5';target.parent.mkdir()
    shutil.copyfile(source,target)
    with h5py.File(target,'r+') as f:
        meta=json.loads(f.attrs['metadata']);meta['seed']=3130;f.attrs['metadata']=json.dumps(meta)
    with pytest.raises(ValueError,match='Duplicate'):
        d.prepare(tmp_path/'cache',[root],require_full=False)


@pytest.mark.parametrize('use_motion',[False,True])
def test_checkpoint_raw_api_and_index_contract(prepared,tmp_path,use_motion):
    model=TrajectoryEncoder(use_motion).eval()
    bundle=dict(schema=d.SCHEMA,window=d.WINDOW,use_motion=use_motion,normalization=prepared.norm,
                sensor_sha256='test_sensor',state_dict=model.state_dict())
    path=tmp_path/'model.pt';torch.save(bundle,path)
    frozen=FrozenTrajectoryEncoder(path)
    source=prepared.episodes[0]['source']
    with h5py.File(source,'r') as f:
        kw=dict(depth=f['tactile/index/depth_m'][:],reference=f['tactile/index/reference_depth_m'][:],
                valid_mask=f['tactile/index/valid_mask'][:],times=f['time_s'][:],sensor_sha256='test_sensor',
                actions=f['action_target'][:],actual_state=f['hand_state'][:,:4])
    got=frozen.encode_window(**kw)
    assert got['quality']=='ok' and len(got['embedding'])==32
    wid=np.flatnonzero((prepared.episode==0)&(prepared.frames[:,-1]==31))
    x,m=prepared.batch_numpy(wid)
    with torch.no_grad(): expected=unit_rows(model(torch.from_numpy(x),torch.from_numpy(m)).numpy())[0]
    np.testing.assert_allclose(got['embedding'],expected,atol=1e-6)
    assert frozen.encode_window(**{**kw,'sensor_sha256':'wrong'})['quality']=='incompatible_sensor'
    assert frozen.encode_window(**{**kw,'depth':kw['depth'][:2],'valid_mask':kw['valid_mask'][:2],'times':kw['times'][:2]})['embedding'] is None
    index=tmp_path/'index.npz';np.savez(index,keys=expected[None],window_ids=np.array([3]))
    d.write_json(str(index)+'.json',{k:got[k] for k in ('encoder_version','sensor_sha256','window')})
    assert search_index(index,got)[0]['window_id']==3
    with pytest.raises(ValueError,match='encoder_version'):
        search_index(index,{**got,'encoder_version':'different'})


def test_views_preserve_order_and_loss_finite():
    model=TrajectoryEncoder()
    x=torch.zeros(4,16,2,24,32);x[:,:,1]=1
    for t in range(16):x[:,t,0]=t
    view=augment(x,torch.Generator().manual_seed(0))
    assert torch.all(torch.diff(view[:,:,0].mean((2,3)),dim=1)>0)
    z=model(view);loss,_=vicreg(z,model(x))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
