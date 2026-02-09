from __future__ import annotations

import argparse
import os

from .experiments import run_ambiguity, run_coverage, run_feasibility_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assembly DSL feasibility prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("run_all", help="Run all experiments")
    p_all.add_argument("--outdir", default="out", help="Output directory")
    p_all.add_argument("--n_samples", type=int, default=200, help="Samples per skeleton")
    p_all.add_argument("--n_param_sets", type=int, default=100, help="Param sets for ambiguity")
    p_all.add_argument("--seed", type=int, default=42)
    default_dataset = os.path.join(os.path.dirname(__file__), "examples", "coverage.csv")
    p_all.add_argument("--dataset", default=default_dataset)

    p_cov = sub.add_parser("coverage", help="Run coverage experiment")
    p_cov.add_argument("--outdir", default="out")
    p_cov.add_argument("--dataset", default=default_dataset)

    p_feas = sub.add_parser("feasibility", help="Run feasibility rate experiment")
    p_feas.add_argument("--outdir", default="out")
    p_feas.add_argument("--n_samples", type=int, default=200)
    p_feas.add_argument("--seed", type=int, default=42)

    p_amb = sub.add_parser("ambiguity", help="Run ambiguity experiment")
    p_amb.add_argument("--outdir", default="out")
    p_amb.add_argument("--n_param_sets", type=int, default=100)
    p_amb.add_argument("--seed", type=int, default=42)

    p_syn = sub.add_parser("synthesize", help="Synthesize DSL from structure + params")
    p_syn.add_argument("--input", required=True, help="Input JSON with structure + params")
    p_syn.add_argument("--outdir", default="out")
    p_syn.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run_all":
        run_coverage(args.dataset, args.outdir)
        run_feasibility_rate(args.outdir, args.n_samples, args.seed)
        run_ambiguity(args.outdir, args.n_param_sets, args.seed)
    elif args.command == "coverage":
        run_coverage(args.dataset, args.outdir)
    elif args.command == "feasibility":
        run_feasibility_rate(args.outdir, args.n_samples, args.seed)
    elif args.command == "ambiguity":
        run_ambiguity(args.outdir, args.n_param_sets, args.seed)
    elif args.command == "synthesize":
        from .synthesize import run_synthesize

        run_synthesize(args.input, args.outdir, args.seed)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
