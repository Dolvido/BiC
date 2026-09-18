"""Pure orchestration mocks: no real dependency, checkpoint, data, tutor or child."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import continuous_tutor_policy as policy

SOURCE = Path(__file__).resolve().parents[1]/"experiments/continuous_tutor_campaign.py"
WORK = dict(author_calls=0, worker_processes=0, compilation_calls=0, preparation_calls=0)


def metrics(joint=20):
    families = {f: dict(counts={m: dict(count=joint if m == "anchor_pair_both" else 20, total=40)
                for m in policy.METRICS}) for f in policy.FAMILIES}
    return dict(by_family=families, overall=dict(counts={m: dict(count=sum(
        families[f]["counts"][m]["count"] for f in policy.FAMILIES), total=120) for m in policy.METRICS}))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.clock = 0.; self.calls = dict.fromkeys(WORK, 0)
        self.author_error = None; self.worker_code = 0; self.worker_unknown = False
        self.worker_seconds = 580.; self.freeze_seconds = 0.; self.worker_hang = False
        self.modules = {}
        def module(name, **items):
            value = ModuleType(name); value.__dict__.update(items); self.modules[name] = value; return value
        helpers = module("experiments.foundation_layout_study", ROOT=self.root, native=Path,
            digest=self.digest, read=self.read, publish=self.publish, encoded=self.encoded,
            relative_root=self.relative, verify_pins=self.verify, utc=lambda: "mock-utc")
        self.data = module("experiments.verified_tutor_cycle_data", prepare_cycle=Mock(side_effect=self.prepare),
            compile_data=Mock(side_effect=self.compile))
        self.author = module("experiments.verified_tutor_author_v2", _json=lambda v:self.encoded(v).decode(),
            author_curriculum=Mock(side_effect=self.author_call))
        self.worker = module("experiments.continuous_tutor_worker", freeze=Mock(side_effect=self.worker_freeze))
        self.transfer = module("experiments.shared_acquisition_transfer_data", load_manifest=Mock(return_value={}))
        module("experiments.verified_tutor_curriculum", _pin=lambda p:len(p)==64,
            contract_sha256=lambda v:hashlib.sha256(self.encoded(v)).hexdigest())
        module("brain_in_computer", __path__=[])
        module("brain_in_computer.curriculum_tutor", LocalTutor=Mock())
        patches = patch.dict(sys.modules, self.modules); patches.start(); self.addCleanup(patches.stop)
        # Explicit package attributes avoid previously imported real dependencies.
        import experiments
        for name, value in self.modules.items():
            if name.startswith("experiments."):
                p=patch.object(experiments,name.split(".")[-1],value,create=True);p.start();self.addCleanup(p.stop)
        spec=importlib.util.spec_from_file_location("mock_campaign_owner",SOURCE)
        self.owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.owner)
        self.owner.time=SimpleNamespace(monotonic=lambda:self.clock,process_time=lambda:self.clock,sleep=self.advance)
        self.owner.subprocess=SimpleNamespace(Popen=Mock(side_effect=self.spawn),STDOUT=-2,CREATE_NO_WINDOW=0)
        self.owner.request_for=Mock(return_value={"request":"mock"})
        self.source=self.record("source.json", {"source":"frozen"})
        self.owner.source_hashes=Mock(return_value={self.source['path']:self.source['sha256']})
        self.evaluation=self.record("banks/manifest.json", {"bank":"immutable"})
        inventory=self.record("inventory/manifest.json", {"inventory":"immutable"})
        self.owner.INVENTORY="inventory";self.owner.INVENTORY_SHA=inventory['sha256']
        self.checkpoint=self.record("parent.json", {"checkpoint":"metadata stub only"})
        self.parent=dict(checkpoint=self.checkpoint,weights_sha256="a"*64,lifetime_updates=2160,
            metrics={r:metrics() for r in policy.ROLES})
        self.teacher=dict(model="mock-model",sha256="b"*64)
        self.spec=policy.cycle_spec(853002001,1)
        self.protection=self.record("protected.json",[])

    def encoded(self,v): return (json.dumps(v,sort_keys=True,separators=(",",":"))+"\n").encode()
    def digest(self,p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(self,p): return json.loads(Path(p).read_bytes())
    def relative(self,p): return str(Path(p).resolve().relative_to(self.root)).replace("\\","/")
    def publish(self,p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:f.write(self.encoded(v))
        return self.digest(p)
    def record(self,name,v):
        path=self.root/name;pin=self.publish(path,v);return dict(path=self.relative(path),sha256=pin)
    def verify(self,pins):
        for name,pin in pins.items():
            if self.digest(self.root/name)!=pin:raise ValueError("mock artifact bytes changed")
    def advance(self,seconds):self.clock+=seconds
    def count(self,name): self.calls[name]+=1;WORK[name]+=1

    def state(self,stage):
        state=dict(schema=self.owner.SCHEMA,stage=stage,cycle=1,completed_cycles=0,
            parent=deepcopy(self.parent),reference=deepcopy(self.parent),retention_reference=deepcopy(self.parent["metrics"]),extra_protected=[self.protection],fresh_only_protected=[],
            decisions=[],physical_totals=dict(worker_updates=0,teacher_chat_attempts=0,teacher_chat_completions=0),
            original_parent_checkpoint=self.checkpoint)
        if stage!='ready':
            cycle=self.root/'seed-cycle';cycle.mkdir()
            self.prepare(cycle);self.calls['preparation_calls']=0
            state.update(cycle_directory=self.relative(cycle),cycle_context=self.owner._record(cycle/'cycle.json'),cycle_spec=self.spec)
            if stage in ('authored','compiled','trained'):
                self.publish(cycle/'author/result.json',dict(teacher_cost=dict(chat_attempts=1,chat_completions=1)))
                state['teacher_decision']={**self.owner._record(cycle/'author/result.json'),'request_sha256':'f'*64}
                state['physical_totals'].update(teacher_chat_attempts=1,teacher_chat_completions=1)
            if stage in ('compiled','trained'):
                self.compile(cycle);self.calls['compilation_calls']=0
                state['compiled']=self.owner._record(cycle/'compiled/manifest.json')
            if stage=='trained':
                record=self.record('saved-worker/summary.json',self.worker_summary(self.spec))
                state['worker']=record;state['physical_totals']['worker_updates']=432
        return state

    def launch(self,name,state,budget):
        directory=self.root/name;directory.mkdir()
        value=dict(schema=self.owner.SCHEMA,seed=853002001,budget_seconds=budget,
            stage_seconds=self.owner.STAGE_SECONDS,final_reserve_seconds=60,teacher=self.teacher,
            initial_state=state,source_sha256=self.owner.source_hashes(),input_sha256={},
            origin_identity={"origin":"fixed"},parent_runtime={"runtime":"fixed"},
            evaluation=dict(directory="banks",manifest_sha256=self.evaluation['sha256']),
            inventory=dict(directory="inventory",manifest_sha256=self.owner.INVENTORY_SHA))
        return directory,self.publish(directory/'launch.json',value)

    def prepare(self,directory,**kwargs):
        self.count('preparation_calls');directory=Path(directory)
        contract=self.record(self.relative(directory/'teaching-contract.json'),{})
        self.publish(directory/'cycle.json',dict(parent={"parent":"bound"},teaching_contract=contract))
        return self.read(directory/'cycle.json')

    def author_call(self,directory,**kwargs):
        self.count('author_calls');self.author_kwargs=kwargs
        self.publish(Path(directory)/'intent.json',dict(request=kwargs['expected_request_sha256']))
        if self.author_error:raise self.author_error
        result=dict(teacher_cost=dict(chat_attempts=1,chat_completions=1))
        pin=self.publish(Path(directory)/'result.json',result);self.advance(1)
        return dict(result=result,result_sha256=pin,reused=False,invocation_cost={})

    def compile(self,directory,**kwargs):
        self.count('compilation_calls');directory=Path(directory)
        fresh=self.record(self.relative(directory/'compiled/fresh.json'),[])
        value=dict(treatment={"different_teaching_bundles":54},fresh_transcripts=fresh)
        self.publish(directory/'compiled/manifest.json',value);self.advance(1);return value

    def worker_summary(self,spec):
        work=dict(synchronized_optimizer_updates=108,retained_episodes=10368,unknown_optimizer_outcomes=0,
            family_episode_exposures={f:3456 for f in policy.FAMILIES})
        branch=dict(status='completed',parent_checkpoint_sha256=self.checkpoint['sha256'],recipe_sha256='d'*64,
            cycle_spec=spec,evaluation_inputs={r:str(i+1)*64 for i,r in enumerate(policy.ROLES)},
            common_withdrawal_sha256='e'*64,actual_work={p:deepcopy(work) for p in ('teaching','withdrawal')},
            teaching_metrics={r:metrics() for r in policy.CORE_ROLES},final_metrics={r:metrics() for r in policy.ROLES})
        tutor=deepcopy(branch);tutor['final_metrics']['dev']=metrics(22)
        continuations={}
        for arm,b in (('procedural',branch),('tutor',tutor)):
            path=self.root/(arm+'-checkpoint.json')
            if not path.exists():self.publish(path,{'arm':arm})
            continuations[arm]=dict(checkpoint=self.owner._record(path),weights_sha256=('6' if arm=='tutor' else '7')*64,
                lifetime_updates=2376,metrics=b['final_metrics'])
        return dict(status='completed',partial_work_unknown=self.worker_unknown,
            branches=dict(procedural=branch,tutor=tutor),continuations=continuations)

    def worker_freeze(self,directory,**kwargs):
        self.worker_kwargs=kwargs;self.worker_dir=Path(directory)
        self.advance(self.freeze_seconds)
        return self.publish(self.worker_dir/'launch.json',{'worker':'frozen mock'})

    def spawn(self,command,**kwargs):
        self.count('worker_processes');self.command=command
        self.publish(self.worker_dir/'execution/summary.json',self.worker_summary(self.worker_kwargs['cycle_spec']))
        self.advance(self.worker_seconds)
        if self.worker_hang:
            process=SimpleNamespace(pid=12345,returncode=None)
            def terminate():process.returncode=-15
            process.poll=lambda:process.returncode
            process.terminate=Mock(side_effect=terminate);process.kill=Mock(side_effect=terminate)
            process.wait=Mock(side_effect=lambda timeout:process.returncode)
            self.hung_process=process
            return process
        return SimpleNamespace(pid=12345,returncode=self.worker_code,poll=lambda:self.worker_code)

    def run_owner(self,directory,pin):
        with patch('sys.stdout',new=io.StringIO()):return self.owner.run(directory,launch_sha256=pin)

    def resume(self,name,prior,budget=660):
        output=self.root/name
        pin=self.owner.freeze(output,parent_directory=prior,
            parent_summary_sha256=self.digest(prior/'execution/summary.json'),
            evaluation_directory=self.root/'banks',evaluation_manifest_sha256=self.evaluation['sha256'],
            teacher=self.teacher,seed=853002001,budget_seconds=budget,resume=True)
        return output,pin

    def test_budget_defers_every_pending_stage_without_operations(self):
        for stage in self.owner.STAGES:
            with self.subTest(stage=stage):
                # Each subcase needs distinct predecessor files.
                state=self.state(stage)
                directory,pin=self.launch('budget-'+stage,state,self.owner.STAGE_SECONDS[stage]+59)
                before=deepcopy(self.calls);result=self.run_owner(directory,pin)
                self.assertEqual(result['final_stage'],stage);self.assertEqual(result['status'],'completed')
                self.assertEqual(self.calls,before)
                if stage!='ready':
                    # Rename fixture directory after its completed test; no active files.
                    (self.root/'seed-cycle').rename(self.root/('fixture-'+stage))

    def test_authentication_cost_can_exhaust_admitted_stage(self):
        first,pin=self.launch('first',self.state('ready'),180)
        calls=[0];sources=self.owner.source_hashes()
        def hash_sources():
            calls[0]+=1
            if calls[0]==2:self.advance(1)
            return sources
        self.owner.source_hashes=Mock(side_effect=hash_sources)
        result=self.run_owner(first,pin)
        self.assertEqual(result['stop_reason'],'wall_budget_after_authentication')
        self.assertEqual(result['stages'],[])
        self.assertEqual(sum(self.calls.values()),0)

    def test_resume_reuses_saved_author_and_absolute_deadline(self):
        directory,pin=self.launch('first',self.state('prepared'),120)
        result=self.run_owner(directory,pin)
        self.assertEqual(result['final_stage'],'authored');self.assertEqual(self.calls['author_calls'],1)
        self.assertEqual(self.author_kwargs['deadline'],60)
        second,pin=self.resume('second',directory)
        result=self.run_owner(second,pin)
        self.assertEqual(result['final_stage'],'compiled');self.assertEqual(self.calls['author_calls'],1)
        self.assertEqual(self.calls['compilation_calls'],1);self.assertEqual(self.calls['worker_processes'],0)
        self.assertEqual(result['new_committed_physical_work']['teacher_chat_attempts'],0)

    def test_completed_worker_reused_for_bound_selection(self):
        first,pin=self.launch('first',self.state('compiled'),660)
        result=self.run_owner(first,pin)
        self.assertEqual(result['final_stage'],'trained');self.assertEqual(self.calls['worker_processes'],1)
        self.assertIn('--deadline',self.command)
        self.assertEqual(self.command[:5],[self.owner.sys.executable,'-B','-m','experiments.continuous_tutor_worker','run'])
        second,pin=self.resume('second',first,budget=120)
        result=self.run_owner(second,pin)
        self.assertEqual(result['completed_cycles'],1);self.assertEqual(result['final_stage'],'ready')
        self.assertEqual(result['parent']['weights_sha256'],'6'*64)
        self.assertEqual(self.calls['worker_processes'],1);self.assertEqual(self.calls['author_calls'],0)
        self.assertEqual(result['new_committed_physical_work']['worker_updates'],0)
        # Rejected branch adoption preserves both the old learner and spent work.
        saved=self.read(first/'execution/summary.json')['last_state']
        retained=self.read(self.root/saved['path'])
        retained['retention_reference']['retention']=metrics(24)
        third,pin=self.launch('retention-stop',retained,120)
        stopped=self.run_owner(third,pin)
        self.assertEqual(stopped['parent'],self.parent)
        self.assertEqual(stopped['parent']['lifetime_updates'],2160)
        self.assertEqual(stopped['physical_totals']['worker_updates'],432)
        state=self.read(self.root/stopped['last_state']['path'])
        decision=self.read(self.root/state['decisions'][-1]['path'])
        self.assertEqual(decision['selected_arm'],'parent')
        self.assertFalse(decision['retention_guard']['passed'])
        self.assertEqual(state['retention_reference'],retained['retention_reference'])
        self.assertEqual(self.calls['worker_processes'],1)


    def test_unknown_author_is_not_retried_or_resumed(self):
        first,pin=self.launch('first',self.state('prepared'),120)
        self.author_error=TimeoutError('mock interrupted request')
        with self.assertRaises(TimeoutError):self.run_owner(first,pin)
        result=self.read(first/'execution/summary.json')
        self.assertTrue(result['partial_work_unknown']);self.assertEqual(result['status'],'failed')
        with self.assertRaises(FileExistsError):self.run_owner(first,pin)
        with self.assertRaises(ValueError):self.resume('second',first)
        self.assertEqual(self.calls['author_calls'],1)

    def test_failed_worker_is_not_retried_or_resumed(self):
        first,pin=self.launch('first',self.state('compiled'),660);self.worker_code=1
        with self.assertRaises(RuntimeError):self.run_owner(first,pin)
        result=self.read(first/'execution/summary.json')
        self.assertTrue(result['partial_work_unknown']);self.assertEqual(result['status'],'failed')
        with self.assertRaises(ValueError):self.resume('second',first)
        with self.assertRaises(FileExistsError):self.run_owner(first,pin)
        self.assertEqual(self.calls['worker_processes'],1)

    def test_foreign_parent_selection_is_refused(self):
        state=self.state('trained');summary=self.read(self.root/state['worker']['path'])
        summary['branches']['tutor']['parent_checkpoint_sha256']='f'*64
        state['worker']=self.record('foreign-worker.json',summary)
        first,pin=self.launch('first',state,120)
        with self.assertRaises(ValueError):self.run_owner(first,pin)
        result=self.read(first/'execution/summary.json')
        self.assertEqual(result['completed_cycles'],0);self.assertEqual(result['parent'],self.parent)

    def test_setup_cannot_reset_admitted_worker_allowance(self):
        first,pin=self.launch('first',self.state('compiled'),660);self.freeze_seconds=600
        with self.assertRaises(TimeoutError):self.run_owner(first,pin)
        self.assertEqual(self.calls['worker_processes'],0)

    def test_worker_watchdog_bounds_cleanup_and_preserves_unknown(self):
        first,pin=self.launch('first',self.state('compiled'),660)
        self.worker_hang=True;self.worker_seconds=0
        with self.assertRaises(TimeoutError):self.run_owner(first,pin)
        result=self.read(first/'execution/summary.json')
        self.assertTrue(result['partial_work_unknown'])
        self.assertEqual(result['final_stage'],'compiled')
        self.hung_process.terminate.assert_called_once()
        self.hung_process.wait.assert_called_once_with(timeout=5)
        self.assertFalse(result['worker_processes'][0]['watchdog']['still_running'])
        self.assertLessEqual(result['wall_seconds'],606)
        with self.assertRaises(ValueError):self.resume('second',first)

    def test_freeze_authenticates_parent_commit_banks_and_sources(self):
        prior=self.root/'prior';prior.mkdir();curriculum=dict(cursor=2160)
        old=dict(schema='bic-sustained-shared-acquisition-v1',source_sha256=self.owner.source_hashes(),
            origin_identity={'origin':'fixed'},expected_runtime={'runtime':'fixed'},data_manifest_sha256=self.evaluation['sha256'])
        launch=self.publish(prior/'launch.json',old)
        scores={r:dict(metrics=metrics()) for r in policy.ROLES}
        commit=self.record('prior/commit.json',dict(launch_sha256=launch,curriculum=curriculum,
            checkpoint=self.checkpoint,scores=scores))
        summary=dict(status='completed',partial_work_unknown=False,launch_sha256=launch,
            last_commit=commit,curriculum=curriculum,evaluations={'2160':scores})
        pin=self.publish(prior/'execution/summary.json',summary)
        self.owner._metadata=Mock(return_value={k:deepcopy(self.parent[k]) for k in ('checkpoint','weights_sha256','lifetime_updates')})
        self.owner._initial_protection=Mock(return_value=(self.protection,self.protection,{},dict(archive_loads=0)))
        def freeze(name,evalpin=None):
            return self.owner.freeze(self.root/name,parent_directory=prior,parent_summary_sha256=pin,
                evaluation_directory=self.root/'banks',evaluation_manifest_sha256=evalpin or self.evaluation['sha256'],teacher=self.teacher)
        self.assertEqual(len(freeze('valid')),64)
        self.assertEqual(self.owner._metadata.call_count,1)
        with self.assertRaises(ValueError):freeze('foreign-bank','f'*64)
        # Parent summary is caller pinned, so altered bytes are refused before metadata.
        (prior/'execution/summary.json').write_text('{}')
        with self.assertRaises(ValueError):freeze('changed-summary')
        self.assertEqual(self.owner._metadata.call_count,1)


if __name__=='__main__':unittest.main()
