"""Additive v2 FedAvg runner. Never opens the frozen test split."""
import argparse
import json
from pathlib import Path

from src.experiment_data import read_json
from src.federated_rounds import run_from_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-round', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    allowed = (root / 'output/experiments/v2_fedavg').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error('Output must be a new run directory under output/experiments/v2_fedavg/')
    config_path = Path(args.config).resolve()
    if not config_path.is_relative_to(root / 'configs/experiments/v2_fedavg'):
        parser.error('Configuration must be under configs/experiments/v2_fedavg/')
    try:
        result = run_from_paths(read_json(config_path), args.data_dir, output,
                                resume=args.resume, stop_after_round=args.stop_after_round)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'FEDAVG ERROR: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
