import argparse
import json
import sys
from pathlib import Path

from .pipeline import predict, run


def main():
    parser = argparse.ArgumentParser(description="VACV SI: organizer-only, SMILES-only regression")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train", "tune", "improve", "advance"):
        command = commands.add_parser(name)
        default_config = {"tune": "configs/nested.json", "improve": "configs/kernel.json", "advance":"configs/advanced.json"}.get(name, "configs/baseline.json")
        command.add_argument("--config", default=default_config)
        command.add_argument("--output", required=True, help="New directory; existing runs are not overwritten")
    command = commands.add_parser("predict")
    command.add_argument("--model", required=True)
    command.add_argument("--test", required=True)
    command.add_argument("--sample", required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "predict":
        result = predict(args.model, args.test, args.sample, args.output)
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        from .target_status import target_status
        review = target_status({'dataset_version':config['dataset_version'],
                                'inputs':{'train':{'sha256':config.get('input_sha256',{}).get('train')}}})
        print('Статус меток: '+review['message'], file=sys.stderr)
        if args.command == "advance":
            from .advanced_experiment import advanced_run
            result = advanced_run(config,args.output)
        elif args.command == "improve":
            from .robust_experiment import robust_run
            result = robust_run(config, args.output)
        elif args.command == "tune":
            from .experiment import nested_run
            result = nested_run(config, args.output)
        else:
            result = run(config, args.output, audit_only=args.command == "audit")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
