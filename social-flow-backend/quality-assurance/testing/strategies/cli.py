"""
CLI for interacting with the strategies package.

Usage examples:
    python -m quality_assurance.testing.strategies.cli plan --target PR-123 --impact 80 --likelihood 70
    python -m quality_assurance.testing.strategies.cli enforce --unit 92 --integration 81
"""

import argparse
import sys
from .test_plan import TestPlan, TestCaseEntry
from .risk_assessment import RiskAssessment
from .ci_policy import CIPolicy
from .unit_strategy import UnitStrategy
from .integration_strategy import IntegrationStrategy
from .utils import safe_dumps


def build_parser():
    p = argparse.ArgumentParser(prog="qa-strategies")
    sub = p.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Create a test plan with risk assessment")
    plan.add_argument("--target", required=True, help="Target identifier (PR, feature, release)")
    plan.add_argument("--impact", type=int, required=True, help="Impact score 0-100")
    plan.add_argument("--likelihood", type=int, required=True, help="Likelihood score 0-100")
    plan.add_argument("--case", action="append", help="Add a test case: id|title|level|minutes", default=[])

    enforce = sub.add_parser("enforce", help="Run policy enforcement locally")
    enforce.add_argument("--unit", type=float, help="Unit coverage percent")
    enforce.add_argument("--integration", type=float, help="Integration coverage percent")
    enforce.add_argument("--impact", type=int, help="Impact score 0-100")
    enforce.add_argument("--likelihood", type=int, help="Likelihood score 0-100")

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        ra = RiskAssessment(impact=args.impact, likelihood=args.likelihood)
        plan = TestPlan(target=args.target, risk=ra)
        for c in args.case:
            # parse case string id|title|level|minutes
            parts = c.split("|")
            if len(parts) < 4:
                print("Invalid case format; expected id|title|level|minutes", file=sys.stderr)
                sys.exit(2)
            entry = TestCaseEntry(id=parts[0], title=parts[1], level=parts[2], estimated_minutes=int(parts[3]))
            plan.add_case(entry)
        print(plan.to_json(pretty=True))

    elif args.cmd == "enforce":
        metrics = {}
        if args.unit is not None:
            metrics["unit"] = args.unit
        if args.integration is not None:
            metrics["integration"] = args.integration
        policy = CIPolicy()
        risk_dict = None
        if args.impact is not None and args.likelihood is not None:
            ra = RiskAssessment(impact=args.impact, likelihood=args.likelihood)
            risk_dict = ra.to_dict()
        result = policy.enforce(metrics=metrics, risk_assessment=risk_dict)
        print(safe_dumps({"success": result.success, "messages": result.messages}, pretty=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
