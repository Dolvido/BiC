"""Postrun training-fit state readout; no policy supervision, update or adoption."""
import argparse
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import time
import traceback

from experiments.foundation_layout_study import ROOT, digest, native, read, publish, verify_pins, relative_root, utc

SCHEMA = "bic-shared-state-fit-diagnostic-v1"
PROTOCOL = "docs/SHARED_STATE_DIAGNOSTIC.md"
PROTOCOL_SHA256 = "369a3ec2318924340331cbd1b3badc5d7d485df62dcfbc4ac12c96afa270bc45"
ARMS, FAMILIES = ("control", "state"), ("color", "count", "switch")
EXPECTED = dict(models=2, checkpoint_loads=2, forwards=18, auxiliary_calls=18,
                episodes=216, state_predictions=25920, target_packs=9, target_rows=108,
                target_turns=1080, updates=0, free_replies=0)


def run(directory, *, summary_sha256, launch_sha256, source_sha256):
    start, cpu = time.monotonic(), time.process_time()
    directory = Path(directory).resolve()
    summary_path, launch_path = directory/"execution/summary.json", directory/"launch.json"
    if (digest(__file__) != source_sha256 or digest(summary_path) != summary_sha256
            or digest(launch_path) != launch_sha256 or digest(ROOT/PROTOCOL) != PROTOCOL_SHA256):
        raise ValueError("explicit diagnostic, protocol, completed summary and launch pins required")
    summary, launch = read(summary_path), read(launch_path)
    if (summary["status"] != "completed" or summary["launch_sha256"] != launch_sha256
            or launch["schema"] != "bic-shared-state-pilot-v1"
            or set(summary["arms"]) != set(ARMS)
            or any(summary["arms"][arm]["cursor"] != 648 for arm in ARMS)):
        raise ValueError("completed 648-update shared-state comparison required")
    pins = {**launch["source_sha256"], relative_root(Path(__file__)):source_sha256, PROTOCOL:PROTOCOL_SHA256,
            relative_root(summary_path):summary_sha256, relative_root(launch_path):launch_sha256}
    verify_pins(pins)
    if any(native(directory/name).exists() for name in ("diagnostic.json", "diagnostic-receipt.json", "diagnostic-started.json")):
        raise FileExistsError("diagnostic is exclusive; existing work must not be replayed")
    receipt = dict(schema=SCHEMA, status="running", started_utc=utc(), input_sha256=pins,
        expected_work=EXPECTED, work=dict.fromkeys(EXPECTED,0), attempted=dict(models=0,checkpoint_loads=0,forwards=0,auxiliary_calls=0),
        automatic_retry=False, automatic_promotion=False, max_seconds=180)
    publish(directory/"diagnostic-started.json", receipt)
    model = None
    try:
        import torch
        from experiments import execution_profile as execution
        from experiments.shared_state_student import SharedStateStudent, build_shared_state_student, ARCHITECTURE
        from experiments.shared_state_targets import pack_state_targets, state_loss
        from experiments.sequence_student import SequenceConfig
        from experiments.composition_data import pack_observations
        from experiments.foundation_evidence import json_digest
        from brain_in_computer.dialogue_student import checkpoint_digest
        from brain_in_computer.language import ByteCodec
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        execution.configure_strict_profile()
        receipt["runtime"] = execution.runtime_profile()
        if receipt["runtime"] != summary["runtime"]:
            raise ValueError("exact original strict FP32 runtime required")
        torch.cuda.reset_peak_memory_stats()

        def boundary():
            if time.monotonic()-start >= 180: raise TimeoutError("diagnostic allowance expired")

        def load(path, pin):
            boundary(); raw = native(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin: raise ValueError("artifact pin differs")
            pins[relative_root(path)] = pin
            return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

        data_directory = ROOT/launch["data_directory"]
        manifest_path = data_directory/"manifest.json"
        if digest(manifest_path) != launch["data_manifest_sha256"]:
            raise ValueError("training-fit manifest changed")
        pins[relative_root(manifest_path)] = launch["data_manifest_sha256"]
        manifest = read(manifest_path)
        banks = load(data_directory/manifest["banks"]["path"], manifest["banks"]["sha256"])
        bank = banks["train_fit"]; del banks
        inventory = manifest["bank_inventory"]["train_fit"]
        if (bank["role"] != "train_fit" or inventory["provenance"] != "reused-actual-training"
                or len(bank["rows"]) != 108 or json_digest(bank["rows"]) != inventory["rows_sha256"]):
            raise ValueError("only the 108 authenticated actual training rows are allowed")
        groups = defaultdict(list)
        for row in bank["rows"]: groups[(row["family"],len(row["turns"]))].append(row)
        if set(groups) != {(family,turns) for family in FAMILIES for turns in (8,10,12)} or any(len(rows)!=12 for rows in groups.values()):
            raise ValueError("nine groups of twelve training rows required")
        config = SequenceConfig(**launch["contract"]["config"])
        prepared = {}
        for key, rows in sorted(groups.items()):
            labels = pack_state_targets(rows)
            inputs = pack_observations([[turn["text"] for turn in row["turns"]] for row in rows],
                max_turns=config.max_turns, max_input_bytes=config.max_input_bytes, max_context_tokens=config.max_positions)
            prepared[key] = (inputs,labels,[row["id"] for row in rows])
            receipt["work"]["target_packs"] += 1; receipt["work"]["target_rows"] += len(rows)
            receipt["work"]["target_turns"] += len(rows)*key[1]
        del bank, groups
        result = dict(schema=SCHEMA, launch_sha256=launch_sha256, summary_sha256=summary_sha256, arms={},
            scope="Training-fit auxiliary diagnostic only; labels never enter native forward; no development/audit targets or adoption gate.")
        for arm in ARMS:
            boundary(); record = summary["checkpoints"][arm]["648"]
            checkpoint_path = ROOT/record["path"]
            artifact_name = checkpoint_path.relative_to(directory/"execution").as_posix()
            if summary["artifact_sha256"][artifact_name] != record["sha256"]:
                raise ValueError("checkpoint receipt hashes disagree")
            receipt["attempted"]["checkpoint_loads"] += 1
            saved = load(checkpoint_path,record["sha256"]); receipt["work"]["checkpoint_loads"] += 1
            if (saved["schema"] != launch["schema"] or saved["launch_sha256"] != launch_sha256
                    or saved["arm"] != arm or saved["step"] != 648 or saved["architecture"] != ARCHITECTURE
                    or saved["auxiliary_weight"] != launch["contract"]["auxiliary_weights"][arm]
                    or saved["learner"]["cursor"] != 648 or saved["learner"]["recipe"]["config"] != launch["contract"]["config"]
                    or saved["runtime"] != receipt["runtime"]):
                raise ValueError("exact final checkpoint identity required")
            receipt["attempted"]["models"] += 1
            model = build_shared_state_student(launch["contract"]["seed"],device="cuda:0",config=config)
            receipt["work"]["models"] += 1
            if type(model) is not SharedStateStudent: raise TypeError("exact model type required")
            model.load_state_dict(saved["learner"]["weights"],strict=True); model.eval()
            before = checkpoint_digest(model)
            if before != saved["weights_sha256"] or any(p.dtype!=torch.float32 for p in model.parameters()):
                raise ValueError("restored full FP32 weights differ")
            del saved
            totals = {key:dict(correct=0,total=0) for key in ("exact","known_exact","unknown_exact")}
            arm_result = dict(checkpoint=record,weights_sha256=before,groups={},by_family={},overall=None)
            family_loss, family_turns = defaultdict(float), defaultdict(int)
            with torch.inference_mode():
                for (family, turns),(inputs,labels,ids) in prepared.items():
                    boundary(); execution.assert_strict_profile()
                    device_inputs = {name:value.to("cuda:0") for name,value in inputs.items()}
                    bos = torch.full((12,turns,1),ByteCodec.BOS,dtype=torch.long,device="cuda:0")
                    receipt["attempted"]["forwards"] += 1
                    output = model(**device_inputs,decoder_input_ids=bos)
                    receipt["work"]["forwards"] += 1; receipt["work"]["episodes"] += 12
                    receipt["attempted"]["auxiliary_calls"] += 1
                    logits = model.state_logits(output["context_states"],device_inputs["eos_positions"])
                    receipt["work"]["auxiliary_calls"] += 1
                    balanced = float(state_loss(logits,labels)); predicted = logits.argmax(-1).cpu()
                    correct, known = predicted.eq(labels), labels.ne(0)
                    counts = {"exact":dict(correct=int(correct.sum()),total=labels.numel()),
                        "known_exact":dict(correct=int((correct&known).sum()),total=int(known.sum())),
                        "unknown_exact":dict(correct=int((correct&~known).sum()),total=int((~known).sum()))}
                    family_counts = arm_result["by_family"].setdefault(family,{key:dict(correct=0,total=0) for key in counts})
                    for key, values in counts.items():
                        for field, value in values.items(): totals[key][field]+=value; family_counts[key][field]+=value
                    family_loss[family] += balanced*12*turns; family_turns[family] += 12*turns
                    arm_result["groups"][f"{family}/{turns}"] = dict(episode_ids=ids,counts=counts,balanced_loss=balanced,
                        targets=labels.tolist(),predictions=predicted.tolist())
                    receipt["work"]["state_predictions"] += labels.numel()
                    del output, logits, device_inputs, bos
            torch.cuda.synchronize(); execution.assert_strict_profile()
            if checkpoint_digest(model) != before or any(m.training for m in model.modules()) or any(p.grad is not None for p in model.parameters()):
                raise ValueError("model state/mode/gradients changed during inference")
            for values in [totals,*arm_result["by_family"].values()]:
                for count in values.values(): count["rate"] = count["correct"]/count["total"] if count["total"] else None
            arm_result["overall"] = dict(counts=totals,balanced_loss=sum(family_loss.values())/sum(family_turns.values()))
            for family in FAMILIES:
                arm_result["by_family"][family] = dict(counts=arm_result["by_family"][family],
                    balanced_loss=family_loss[family]/family_turns[family])
            arm_result["model_unchanged"] = True; result["arms"][arm] = arm_result
            del model; model = None
        if receipt["work"] != EXPECTED: raise ValueError("fixed diagnostic work differs")
        verify_pins(pins)
        receipt["diagnostic_sha256"] = publish(directory/"diagnostic.json",result)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize(); receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        except BaseException as error:
            receipt.update(status="failed",cleanup_error=repr(error))
        receipt.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,input_sha256=pins)
        publish(directory/"diagnostic-receipt.json",receipt)
    return dict(status=receipt["status"],diagnostic_sha256=receipt["diagnostic_sha256"],work=receipt["work"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    for name in ("summary-sha256","launch-sha256","source-sha256"): parser.add_argument("--"+name,required=True)
    args = parser.parse_args()
    print(run(args.directory,summary_sha256=args.summary_sha256,launch_sha256=args.launch_sha256,source_sha256=args.source_sha256))
