"""Command-line interface for post-training-toolkit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def diagnose_cli():
    """Run diagnostics on a training run."""
    parser = argparse.ArgumentParser(
        description="Run diagnostics on RLHF training logs",
        prog="ptt-diagnose",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to metrics.jsonl or run directory",
    )
    parser.add_argument(
        "--reports-dir", "-o",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate diagnostic plots",
    )
    parser.add_argument(
        "--trainer-type",
        type=str,
        default=None,
        choices=["dpo", "ppo", "sft", "orpo", "kto", "cpo", "grpo"],
        help="Override auto-detected trainer type",
    )
    
    args = parser.parse_args()
    
    from post_training_toolkit.models import run_diagnostics
    
    try:
        report_path = run_diagnostics(
            args.input,
            args.reports_dir,
            make_plots=args.make_plots,
            trainer_type=args.trainer_type,
        )
        print(f"✓ Report generated: {report_path}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


def compare_cli():
    """Compare checkpoints and recommend the best one."""
    parser = argparse.ArgumentParser(
        description="Compare checkpoints and get recommendation",
        prog="ptt-compare",
    )
    parser.add_argument(
        "--run-dir", "-d",
        type=Path,
        required=True,
        help="Path to run directory with snapshots",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file for recommendation (default: stdout)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    
    args = parser.parse_args()
    
    from post_training_toolkit.models.checkpoints import recommend_checkpoint
    import json
    
    try:
        recommendation = recommend_checkpoint(args.run_dir)
        
        if recommendation is None:
            print("✗ Not enough data for checkpoint comparison", file=sys.stderr)
            return 1
        
        if args.format == "json":
            output = json.dumps(recommendation.to_dict(), indent=2)
        else:
            output = f"""
Checkpoint Recommendation
========================
Recommended: Step {recommendation.step}

Justification:
{recommendation.justification}

Candidates (ranked by score):
"""
            for i, c in enumerate(recommendation.candidates[:5], 1):
                output += f"  {i}. Step {c.step}: score={c.overall_score:.3f}, drift={c.drift_score:.3f}, refusal={c.refusal_rate:.1%}\n"
        
        if args.output:
            args.output.write_text(output)
            print(f"✓ Recommendation written to: {args.output}")
        else:
            print(output)
        
        return 0
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


def validate_resume_cli():
    """Validate a training run resume."""
    parser = argparse.ArgumentParser(
        description="Validate resume from checkpoint",
        prog="ptt-validate-resume",
    )
    parser.add_argument(
        "--run-dir", "-d",
        type=Path,
        required=True,
        help="Path to run directory",
    )
    parser.add_argument(
        "--checkpoint-step", "-s",
        type=int,
        required=True,
        help="Step number of checkpoint being resumed from",
    )
    parser.add_argument(
        "--checkpoint-path", "-p",
        type=Path,
        default=None,
        help="Path to checkpoint directory (optional, for integrity check)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    
    args = parser.parse_args()
    
    from post_training_toolkit.models.checkpoints import validate_resume
    import json
    
    try:
        result = validate_resume(
            args.run_dir,
            args.checkpoint_step,
            str(args.checkpoint_path) if args.checkpoint_path else None,
        )
        
        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "✓ VALID" if result.is_valid else "✗ INVALID"
            print(f"""
Resume Validation: {status}
===========================
Resuming from step: {result.resumed_from_step}
Expected next step: {result.expected_next_step}
""")
            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  ⚠ {w}")
            if result.errors:
                print("Errors:")
                for e in result.errors:
                    print(f"  ✗ {e}")
        
        return 0 if result.is_valid else 1
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Default to diagnose if run directly
    sys.exit(diagnose_cli())


def agent_diagnose_cli():
    """Run diagnostics on agent traces."""
    parser = argparse.ArgumentParser(
        description="Run diagnostics on agent trace logs",
        prog="ptt-agent-diagnose",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to agent traces JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file for report (default: stdout)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Cost budget per episode (for budget alerts)",
    )
    parser.add_argument(
        "--export-dpo",
        type=Path,
        default=None,
        help="Export preference pairs dataset to this path (Parquet format)",
    )
    
    args = parser.parse_args()
    
    from post_training_toolkit.agents import AgentRunLog, analyze_runs, to_preference_pairs
    import json
    
    try:
        # Load traces
        runs = AgentRunLog.from_jsonl(args.input)
        print(f"Loaded {len(runs)} episodes from {args.input}", file=sys.stderr)
        
        # Run diagnostics
        report = analyze_runs(runs, budget_per_episode=args.budget)
        
        # Output report
        if args.format == "json":
            output = json.dumps({
                "total_episodes": report.total_episodes,
                "success_rate": report.success_rate,
                "avg_steps": report.avg_steps,
                "avg_tokens": report.avg_tokens,
                "total_cost": report.total_cost,
                "tool_error_rate": report.tool_error_rate,
                "episodes_with_loops": report.episodes_with_loops,
                "episodes_with_tool_errors": report.episodes_with_tool_errors,
                "insights": [
                    {
                        "type": i.type,
                        "severity": i.severity,
                        "message": i.message,
                        "episodes": i.episodes,
                        "data": i.data,
                    }
                    for i in report.insights
                ],
            }, indent=2)
        else:
            output = str(report)
        
        if args.output:
            args.output.write_text(output)
            print(f"✓ Report written to: {args.output}", file=sys.stderr)
        else:
            print(output)
        
        # Export DPO dataset if requested
        if args.export_dpo:
            try:
                dataset = to_preference_pairs(
                    runs,
                    positive=lambda e: e.success is True and e.total_steps < 15,
                    negative=lambda e: e.success is False or e.has_repeated_tool_pattern(),
                )
                dataset.to_parquet(str(args.export_dpo))
                print(f"✓ Exported {len(dataset)} preference pairs to: {args.export_dpo}", file=sys.stderr)
            except ValueError as e:
                print(f"⚠ Could not export DPO dataset: {e}", file=sys.stderr)
        
        return 0 if not report.has_critical_issues else 1
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
