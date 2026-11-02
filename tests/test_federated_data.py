"""Stage 1 data-contract tests. Every generated array is a temporary synthetic fixture."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_data import FEATURES, weather_view
from src.federated_data import (ClientTrainDataset, build_client_index, load_training_clients,
                                load_coordinator_validation)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(directory, *, mutate=None, include_validation=True, include_test=True):
    directory.mkdir()
    names = np.array(['node-z', '47', 'node-a'])
    rng = np.random.default_rng(4)
    arrays = {'node_names': names, 'power_baseline_coefficients': np.array([2., .5], dtype=np.float64)}
    for split, first, days in (('train', '2022-01-01T01:00', 12), ('val', '2022-02-12T01:00', 4),
                               ('test', '2022-02-21T01:00', 4)):
        if split == 'val' and not include_validation or split == 'test' and not include_test:
            continue
        ids = np.tile(np.arange(3), days)
        count = len(ids)
        x = rng.normal(size=(count, 6, 14)).astype(np.float32)
        x[:, :, 13] = ids[:, None]
        y = (np.arange(count, dtype=np.float32) / 10)[:, None]
        if split == 'val': y[:] = 777
        if split == 'test': y[:] = 999
        origins = np.repeat(np.datetime64(first, 'ns') + np.arange(days) * np.timedelta64(1, 'D'), 3)
        arrays.update({f'X_{split}':x, f'y_{split}':y, f'node_{split}':names[ids], f'origin_{split}':origins})
    meta = {'feature_names': FEATURES, 'nodes':names.tolist(), 'history_steps':6, 'horizon_steps':6,
            'months':['22-01', '22-02'], 'settings':{'resample_minutes':10,'history_hours':1,'horizon_minutes':60},
            'x_mean':[.6]*13+[0.], 'x_std':[.4]*13+[1.], 'y_mean':[.5], 'y_std':[.2],
            'normalization_fit':'pooled training windows only; node indices are not standardized',
            'split_boundaries_utc':{'train_end_val_start':'2022-02-11T07:10:00+00:00',
                                    'val_end_test_start':'2022-02-20T03:30:00+00:00'}}
    if mutate is not None: mutate(arrays, meta)
    inventory = {}
    for name,value in arrays.items():
        path = directory/f'{name}.npy'
        with path.open('xb') as handle:np.save(handle,value,allow_pickle=False)
        inventory[path.name] = {'sha256':sha(path),'shape':list(value.shape),'dtype':str(value.dtype)}
    (directory/'preprocessing.json').write_text(json.dumps(meta))
    contract = {'version':1, 'arrays':inventory, 'metadata_sha256':sha(directory/'preprocessing.json'),
                'feature_names':FEATURES, 'scalers':{k:meta[k] for k in ('x_mean','x_std','y_mean','y_std')},
                'chronological_cuts':meta['split_boundaries_utc']}
    (directory/'contract.json').write_text(json.dumps(contract))
    return sha(directory/'contract.json'),arrays,meta


def clients(directory, digest, **kwargs):
    return load_training_clients(directory,expected_contract_sha256=digest,minimum_windows=4,minimum_days=2,**kwargs)


class PartitionTests(unittest.TestCase):
    def test_interleaved_membership_disjoint_exhaustive_and_in_frozen_node_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,arrays,_=fixture(path)
            registry=clients(path,digest)
            self.assertEqual(registry.client_ids,('node-z','47','node-a'))
            self.assertEqual(registry.num_windows,36)
            rows=np.concatenate([p.row_indices for p in registry.partitions])
            np.testing.assert_array_equal(np.sort(rows),np.arange(36))
            self.assertEqual(len(set(rows.tolist())),36)
            for i,p in enumerate(registry.partitions):
                self.assertEqual(p.node_index,i)
                self.assertEqual(p.distinct_training_days,12)
                np.testing.assert_array_equal(p.row_indices,np.arange(i,36,3))
                np.testing.assert_array_equal(arrays['node_train'][p.row_indices],p.client_id)
                with self.assertRaises(ValueError):p.row_indices.setflags(write=True)
                with self.assertRaises(FrozenInstanceError):p.client_id='changed'
            with self.assertRaises(TypeError):registry.training_counts['47']=1

    def test_training_loader_never_opens_validation_or_test(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,_,_=fixture(path)
            original=np.load;opened=[]
            def train_only(filename,*args,**kwargs):
                name=Path(filename).name;opened.append(name)
                if '_val.' in name or '_test.' in name:raise AssertionError('Worker loader opened held-out arrays')
                return original(filename,*args,**kwargs)
            with mock.patch('src.federated_data.np.load',side_effect=train_only):
                registry=clients(path,digest)
                dataset=registry.client('47')
                dataset.batch([0,1])
            self.assertEqual(set(opened),{'node_names.npy','power_baseline_coefficients.npy',
                                         'X_train.npy','y_train.npy','origin_train.npy','node_train.npy'})

    def test_training_loader_works_without_any_validation_or_test_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,_,_=fixture(path,include_validation=False,include_test=False)
            self.assertEqual(clients(path,digest).num_windows,36)

    def test_client_only_carries_its_own_training_copies_and_survives_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,arrays,_=fixture(path)
            dataset=clients(path,digest).client('47')
            self.assertFalse(hasattr(dataset,'split'))
            self.assertFalse(hasattr(dataset,'array'))
            self.assertFalse(hasattr(dataset,'path'))
            self.assertFalse(any(isinstance(v,Path) for v in vars(dataset).values()))
            for value in vars(dataset).values():
                if isinstance(value,np.ndarray):
                    self.assertEqual(value.shape[0],12)
                    self.assertFalse(isinstance(value,np.memmap))
            rebuilt=pickle.loads(pickle.dumps(dataset))
            rows=np.arange(1,36,3)
            np.testing.assert_array_equal(rebuilt.batch(np.arange(12)).targets,arrays['y_train'][rows])
            self.assertFalse((rebuilt.batch(np.arange(12)).targets==777).any())
            self.assertFalse((rebuilt.batch(np.arange(12)).targets==999).any())
            with mock.patch('src.federated_data.np.load',side_effect=AssertionError('Worker performed file I/O')):
                rebuilt[0];rebuilt.batch([0,1])
            with self.assertRaises(TypeError):ClientTrainDataset(arrays['X_val'],arrays['y_val'])

    def test_weather_mask_preserves_all_other_data_and_scalers_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,arrays,meta=fixture(path)
            before={p.name:sha(p) for p in path.iterdir()}
            masked=clients(path,digest).client('47')
            enabled=clients(path,digest,weather='enabled').client('47')
            rows=np.arange(1,36,3);m=masked.batch(np.arange(12));e=enabled.batch(np.arange(12))
            self.assertEqual(m.inputs.shape,(12,6,14))
            np.testing.assert_array_equal(m.inputs,weather_view(arrays['X_train'][rows],'masked'))
            np.testing.assert_array_equal(e.inputs,arrays['X_train'][rows])
            np.testing.assert_array_equal(m.targets,e.targets)
            np.testing.assert_array_equal(m.origins,e.origins)
            np.testing.assert_array_equal(m.row_indices,rows)
            np.testing.assert_array_equal(m.inputs[:,:,13],1)
            self.assertEqual(masked.scalers.x_mean,tuple(meta['x_mean']))
            self.assertEqual(masked.scalers.x_std,tuple(meta['x_std']))
            self.assertEqual(masked.scalers.y_mean,tuple(meta['y_mean']))
            self.assertEqual(masked.scalers.y_std,tuple(meta['y_std']))
            self.assertEqual(masked.baseline_coefficients,(2.,.5))
            with self.assertRaises(FrozenInstanceError):masked.scalers.y_std=(9.,)
            m.inputs[:]=100;m.targets[:]=100
            np.testing.assert_array_equal(masked[0][1],arrays['y_train'][1])
            self.assertEqual(before,{p.name:sha(p) for p in path.iterdir()})

    def test_local_index_bounds_and_node_mapping_are_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,_,_=fixture(path)
            registry=clients(path,digest);dataset=registry.client('47')
            for value in (-1,12,True,1.5):
                with self.subTest(value=value),self.assertRaises(IndexError):dataset[value]
            for values in ([-1],[12],[1.5],[True],[[0]]):
                with self.subTest(values=values),self.assertRaises(IndexError):dataset.batch(values)
            for name in (47,'unknown','0'):
                with self.subTest(name=name),self.assertRaises(KeyError):registry.client(name)

    def test_default_requirements_do_not_silently_filter_a_client(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,_,_=fixture(path)
            with self.assertRaisesRegex(ValueError,'insufficient local training data'):
                load_training_clients(path,expected_contract_sha256=digest)
            with self.assertRaisesRegex(ValueError,'insufficient local training data'):
                load_training_clients(path,expected_contract_sha256=digest,minimum_windows=4,minimum_days=13)
            for value in (0,-1,True,1.5):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    load_training_clients(path,expected_contract_sha256=digest,minimum_windows=value,minimum_days=1)

    def test_later_split_origin_cannot_be_inserted_into_training_even_with_valid_hashes(self):
        for injected in ('2022-02-11T07:10','2022-02-21T01:00','2022-02-11T06:20'):
            with self.subTest(injected=injected),tempfile.TemporaryDirectory() as temp:
                def mutate(a,m):a['origin_train'][0]=np.datetime64(injected,'ns')
                path=Path(temp)/'data';digest,_,_=fixture(path,mutate=mutate)
                with self.assertRaisesRegex(ValueError,'training boundary'):clients(path,digest)

    def test_target_ending_exactly_at_train_cut_is_accepted(self):
        cut='2022-02-11T07:10:00+00:00'
        origins=np.array(['2022-02-11T06:00','2022-02-11T06:10'],dtype='datetime64[ns]')
        parts=build_client_index(np.array(['a','a']),origins,np.array(['a']),train_end_utc=cut,
                                 minimum_windows=1,minimum_days=1)
        self.assertEqual(parts[0].num_examples,2)

    def test_duplicate_unordered_off_grid_nat_and_cross_node_data_are_rejected(self):
        cases={
            'duplicate':lambda a,m:a['origin_train'].__setitem__(3,a['origin_train'][0]),
            'unordered':lambda a,m:a['origin_train'].__setitem__(0,np.datetime64('2022-01-20T01:00','ns')),
            'off_grid':lambda a,m:a['origin_train'].__setitem__(0,np.datetime64('2022-01-01T01:01','ns')),
            'nat':lambda a,m:a['origin_train'].__setitem__(0,np.datetime64('NaT','ns')),
            'cross_node':lambda a,m:a['X_train'].__setitem__((0,-1,13),1),
            'unknown_node':lambda a,m:a['node_train'].__setitem__(0,'absent'),
            'nonfinite':lambda a,m:a['y_train'].__setitem__((0,0),np.nan),
            'early_history':lambda a,m:a['origin_train'].__setitem__(0,np.datetime64('2022-01-01T00:50','ns')),
            'timestamp_overflow':lambda a,m:a['origin_train'].__setitem__(0,np.datetime64('2262-04-11T23:40','ns')),
            'wrong_dtype':lambda a,m:a.__setitem__('X_train',a['X_train'].astype(np.float64)),
            'wrong_shape':lambda a,m:a.__setitem__('X_train',a['X_train'][:,:5]),
        }
        for name,mutate in cases.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'data';digest,_,_=fixture(path,mutate=mutate)
                with self.assertRaises(ValueError):clients(path,digest)

    def test_contract_array_and_metadata_tampering_fail_closed(self):
        for target in ('contract.json','preprocessing.json','node_train.npy','X_train.npy'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'data';digest,_,_=fixture(path)
                with (path/target).open('ab') as handle:handle.write(b'changed')
                with self.assertRaises(ValueError):clients(path,digest)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,_,_=fixture(path)
            for anchor in ('0'*64,'',None,'x'*64):
                with self.subTest(anchor=anchor),self.assertRaises(ValueError):clients(path,anchor)
            with self.assertRaises(ValueError):clients(path,digest,weather='off')

    def test_changed_scalers_coefficients_features_or_mapping_rejected(self):
        cases={
            'negative_std':lambda a,m:m['x_std'].__setitem__(0,-1),
            'node_standardized':lambda a,m:m['x_mean'].__setitem__(13,1),
            'baseline':lambda a,m:a['power_baseline_coefficients'].__setitem__(0,9),
            'mapping':lambda a,m:m['nodes'].__setitem__(0,'other'),
            'history':lambda a,m:m.__setitem__('history_steps',12),
            'duplicate_names':lambda a,m:a['node_names'].__setitem__(1,a['node_names'][0]),
        }
        for name,mutate in cases.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'data';digest,_,_=fixture(path,mutate=mutate)
                with self.assertRaises((ValueError,KeyError)):clients(path,digest)

    def test_coordinator_validation_is_a_separate_capability_and_never_opens_test(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data';digest,arrays,_=fixture(path)
            original=np.load;opened=[]
            def validation_only(filename,*args,**kwargs):
                name=Path(filename).name;opened.append(name)
                if '_train.' in name or '_test.' in name:raise AssertionError('Coordinator opened non-validation examples')
                return original(filename,*args,**kwargs)
            with mock.patch('src.federated_data.np.load',side_effect=validation_only):
                coordinator=load_coordinator_validation(path,expected_contract_sha256=digest)
            for actual,key in zip(coordinator.split('val'),('X_val','y_val','origin_val','node_val')):
                np.testing.assert_array_equal(actual,arrays[key])
                self.assertFalse(actual.flags.writeable)
            for split in ('train','test'):
                with self.assertRaises(PermissionError):coordinator.split(split)
            self.assertEqual(set(opened),{'node_names.npy','power_baseline_coefficients.npy','X_val.npy','y_val.npy','origin_val.npy','node_val.npy'})
            with self.assertRaises(TypeError):coordinator.meta['y_std']=(9.,)

    def test_coordinator_rejects_training_and_test_origins_inside_validation(self):
        for origin in ('2022-01-10T01:00','2022-02-20T03:40'):
            with self.subTest(origin=origin),tempfile.TemporaryDirectory() as temp:
                def mutate(a,m):a['origin_val'][0]=np.datetime64(origin,'ns')
                path=Path(temp)/'data';digest,_,_=fixture(path,mutate=mutate)
                with self.assertRaisesRegex(ValueError,'Validation target'):
                    load_coordinator_validation(path,expected_contract_sha256=digest)


@unittest.skipUnless(os.environ.get('M100_FEDAVG_REAL_FROZEN'),'Set M100_FEDAVG_REAL_FROZEN for training-only M100 integration')
class RealTrainingDataTests(unittest.TestCase):
    def test_all_4506133_windows_form_979_clients_and_keep_frozen_weather_scalers(self):
        path=Path(os.environ['M100_FEDAVG_REAL_FROZEN'])
        digest='24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722'
        original=np.load;opened=[]
        def guarded(filename,*args,**kwargs):
            name=Path(filename).name;opened.append(name)
            if '_val.' in name or '_test.' in name:raise AssertionError('Real-data worker loader accessed held-out arrays')
            return original(filename,*args,**kwargs)
        with mock.patch('src.federated_data.np.load',side_effect=guarded):
            registry=load_training_clients(path,expected_contract_sha256=digest)
            self.assertEqual(len(registry.client_ids),979)
            self.assertEqual(registry.num_windows,4506133)
            membership=np.concatenate([part.row_indices for part in registry.partitions])
            np.testing.assert_array_equal(np.sort(membership),np.arange(4506133))
            self.assertEqual(sum(registry.training_counts.values()),4506133)
            for name in ('1','47','979'):
                data=registry.client(name);batch=data.batch([0,len(data)-1])
                expected=weather_view(original(path/'X_train.npy',mmap_mode='r')[batch.row_indices].copy(),'masked')
                np.testing.assert_array_equal(batch.inputs,expected)
                np.testing.assert_array_equal(batch.targets,original(path/'y_train.npy',mmap_mode='r')[batch.row_indices])
                np.testing.assert_array_equal(batch.origins,original(path/'origin_train.npy',mmap_mode='r')[batch.row_indices])
            self.assertEqual(registry.client('47').node_index,46)
            meta=json.loads((path/'preprocessing.json').read_text())
            self.assertEqual(registry.scalers.x_std,tuple(meta['x_std']))
            self.assertEqual(registry.scalers.y_std,tuple(meta['y_std']))
        self.assertTrue(all('_val.' not in name and '_test.' not in name for name in opened))


if __name__=='__main__':unittest.main()
