"""Run an explicitly pinned curriculum owner within a local compute allowance.

This entry point never discovers a latest checkpoint, creates a learner, selects
an architecture, calls a teacher or promotes weights. Keep the returned pin in
caller-controlled storage; an editable neighboring receipt is not a trust root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import torch

from experiments.execution_profile import configure_strict_profile
from experiments.foundation_cycle_owner import CycleOwner


SCHEMA = "bic-foundation-cycle-invocation-cli-v1"


def _nonnegative(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("a nonnegative integer is required") from error
    if str(result) != str(value) or not 0 <= result < 2**63:
        raise argparse.ArgumentTypeError("a canonical nonnegative bounded integer is required")
    return result


def _positive_hours(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("positive finite hours are required") from error
    if not math.isfinite(result) or result <= 0 or not math.isfinite(result * 3600):
        raise argparse.ArgumentTypeError("positive finite hours are required")
    return result


def _pin(value):
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("an explicit lowercase SHA256 pin is required")
    return value


def _positive_integer(value):
    result = _nonnegative(value)
    if result == 0:
        raise argparse.ArgumentTypeError("a positive integer is required")
    return result


def _device(value):
    try:
        selected = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise argparse.ArgumentTypeError("an explicit CPU or CUDA device is required") from error
    if selected.type not in ("cpu", "cuda") or str(selected) != value:
        raise argparse.ArgumentTypeError("a canonical CPU or CUDA device is required")
    return value


def _configure_runtime(args):
    # No tensor/model/CUDA operation may precede explicit strict configuration.
    # The subsequent pinned load still requires exact saved runtime equality.
    if torch.device(args.device).type == "cuda" or args.strict_profile:
        configure_strict_profile()
    if args.interop_threads is not None and torch.get_num_interop_threads() != args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.threads)


def _publish(path, value):
    """Publish one complete fsynced JSON image without replacing an existing file."""
    temporary = None
    try:
        raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                prefix=".cycle-result-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--owner", required=True, type=Path)
    result.add_argument("--expected-sha256", required=True, type=_pin)
    result.add_argument("--receipt", required=True, type=Path,
        help="New result path outside the owner directory; never automatically trusted on restart")
    result.add_argument("--device", default="cpu", type=_device,
        help="Exact saved device spelling, for example cpu, cuda or cuda:0")
    result.add_argument("--threads", type=_positive_integer, default=1)
    result.add_argument("--interop-threads", type=_positive_integer,
        help="Explicit saved interop setting; omission keeps the fresh process setting")
    result.add_argument("--strict-profile", action="store_true",
        help="Configure the strict execution profile also on CPU; CUDA always requires it")
    result.add_argument("--max-updates", type=_nonnegative)
    result.add_argument("--max-cycles", type=_nonnegative)
    result.add_argument("--hours", type=_positive_hours,
        help="Wall allowance including load; checked between nonpreemptive operations")
    result.add_argument("--min-generation", type=_nonnegative)
    result.add_argument("--recovery-policy", default="stop", choices=("stop", "acknowledge_unknown"))
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.max_updates is None and args.max_cycles is None and args.hours is None:
        arguments.error("supply --max-updates, --max-cycles or --hours")
    if args.min_generation == 0:
        arguments.error("--min-generation must be at least one")
    owner_path, receipt_path = args.owner.resolve(), args.receipt.resolve()
    if receipt_path.is_relative_to(owner_path.parent):
        arguments.error("--receipt must be outside the owner directory")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve this invocation name before any possible work. A crash leaves its
    # intent visible; the owner independently journals physical learning work.
    intent_path = receipt_path.with_name(receipt_path.name + ".intent.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    started = time.monotonic()
    deadline = None if args.hours is None else started + args.hours * 3600
    if deadline is not None and not math.isfinite(deadline):
        arguments.error("wall deadline exceeds the finite clock range")
    intent = dict(schema=SCHEMA, owner=str(owner_path), expected_sha256=args.expected_sha256,
        requested_device=args.device, recovery_policy=args.recovery_policy,
        runtime_setup=dict(threads=args.threads, interop_threads=args.interop_threads,
            strict_profile=torch.device(args.device).type == "cuda" or args.strict_profile),
        allowances=dict(max_updates=args.max_updates, max_cycles=args.max_cycles, hours=args.hours),
        min_generation=args.min_generation, pid=os.getpid(),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _publish(intent_path, intent)
    owner = None
    result = dict(intent, status="loading", automatic_promotion=False)
    failure = None
    try:
        _configure_runtime(args)
        owner = CycleOwner.load(owner_path, expected_sha256=args.expected_sha256,
            min_generation=args.min_generation, device=args.device, recovery_policy=args.recovery_policy)
        result["load_seconds"] = time.monotonic() - started
        report = owner.run(max_updates=args.max_updates, max_cycles=args.max_cycles, deadline=deadline)
        result.update(status="completed_invocation", run=report, owner_sha256=report["owner_sha256"])
    except BaseException as error:
        failure = error
        result.update(status="failed", error=repr(error), run=None if owner is None else owner.last_report)
        # Do not manufacture a current pin when load/publication/accounting is
        # uncertain. The durable owner's evidence remains available for review.
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["scope"] = ("Prescribed curriculum execution only. A deadline can overrun during load, "
            "admission, scoring or publication. Lost/unmeasured work is governed by owner invocation "
            "receipts. No learning-benefit or general-intellect claim.")
        _publish(receipt_path, result)
    if failure is not None:
        raise failure
    print(json.dumps(dict(status=result["status"], owner_sha256=result["owner_sha256"],
        receipt=str(receipt_path), run_status=result["run"]["status"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
