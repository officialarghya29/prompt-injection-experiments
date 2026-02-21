#!/usr/bin/env python3
"""
Run all 4 experiments for the research paper.

This will take several hours due to LLM generation times.
Results are saved incrementally to the database and JSON files.

Usage:
    python run_experiments.py [--quick]
    
Options:
    --quick  Run with subset of prompts for quick testing (5 attacks, 2 benign)
"""

import sys
import argparse
import logging
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment_runner import ExperimentRunner
# Try to import large benign prompts if they exist
try:
    # Add root to path to ensure data package is findable
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.benign_prompts_large import BENIGN_PROMPTS_LARGE
except ImportError:
    BENIGN_PROMPTS_LARGE = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / "experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def estimate_time(num_attacks, num_benign, num_configs, workers=1):
    """Estimate total experiment time."""
    # Conservative estimates:
    # - Attack prompts: ~10s each (with defenses, some blocked early)
    # - Benign prompts: ~30s each (longer, more complex)
    attack_time = (num_attacks * num_configs * 10) / workers  # seconds
    benign_time = (num_benign * num_configs * 30) / workers  # seconds
    total_seconds = attack_time + benign_time
    
    hours = total_seconds / 3600
    return hours


def main():
    parser = argparse.ArgumentParser(description="Run prompt injection experiments")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with subset of prompts"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific experiment only (1-4)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of randomized trials per configuration (default: 1, recommended: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for trials (default: 1, recommended: 5 for free tier)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt for long-running experiments"
    )
    parser.add_argument(
        "--config-filter",
        type=str,
        help="Filter configurations by name (e.g., 'FullWorkflow')"
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run Utility Stress Test using 1,000 benign prompts"
    )
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PROMPT INJECTION EXPERIMENT RUNNER")
    logger.info("="*80)
    
    # Load stress test prompts if requested
    benign_set = None
    if args.stress_test:
        if BENIGN_PROMPTS_LARGE:
            logger.info(f"STRESS TEST MODE: Loading {len(BENIGN_PROMPTS_LARGE)} benign prompts")
            benign_set = BENIGN_PROMPTS_LARGE
        else:
            logger.error("Large benign dataset not found. Run src/tools/generate_benign_prompts.py first.")
            return 1

    runner = ExperimentRunner(
        n_trials=args.trials, 
        random_seed=args.seed,
        max_workers=args.workers,
        benign_prompts=benign_set
    )
    
    # Quick mode for testing
    if args.quick:
        logger.info("\nQUICK MODE: Using subset of prompts")
        original_attacks = runner.attack_prompts.copy()
        original_benign = runner.benign_prompts.copy()
        
        runner.attack_prompts = dict(list(original_attacks.items())[:5])
        runner.benign_prompts = dict(list(original_benign.items())[:2])
        
        logger.info(
            f"  Attacks: {len(runner.attack_prompts)}/{len(original_attacks)}"
        )
        logger.info(
            f"  Benign: {len(runner.benign_prompts)}/{len(original_benign)}"
        )
    
    # Estimate time
    num_attacks = len(runner.attack_prompts)
    num_benign = len(runner.benign_prompts)
    
    if args.experiment:
        logger.info(f"\nRunning Experiment {args.experiment} only")
        if args.experiment == 1:
            num_configs = 4
        elif args.experiment == 2:
            num_configs = 4  # 4 isolation modes
        elif args.experiment == 3:
            num_configs = 5
        elif args.experiment == 4:
            num_configs = 5
    else:
        # All experiments
        num_configs = 4 + 4 + 5 + 5  # Sum of all configs
    
    estimated_hours = estimate_time(num_attacks, num_benign, num_configs, args.workers) * args.trials
    logger.info(f"\nEstimated time: {estimated_hours:.1f} hours ({estimated_hours * 60:.0f} minutes)")
    logger.info(f"  ({num_attacks} attacks + {num_benign} benign) × {num_configs} configs × {args.trials} trials")
    logger.info(f"  = {(num_attacks * num_configs + num_benign * num_configs) * args.trials} total LLM calls")
    
    if args.trials > 1:
        logger.info(f"\n📊 STATISTICAL MODE: {args.trials} randomized trials per configuration")
        logger.info(f"   Results will include confidence intervals and statistical tests")
    
    # Confirm if not quick mode and not auto-approved
    if not args.quick and not args.yes and estimated_hours > 1.0:
        try:
            response = input(f"\nThis will take approximately {estimated_hours:.1f} hours. Continue? [y/N] ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                return 0
        except EOFError:
            logger.error("Non-interactive mode detected but no --yes flag provided. Aborting.")
            return 1
    
    start_time = time.time()
    
    try:
        if args.experiment:
            # Run specific experiment
            logger.info(f"\nStarting Experiment {args.experiment}...")
            
            if args.experiment == 1:
                results = {"experiment_1": runner.run_experiment_1()}
            elif args.experiment == 2:
                results = {"experiment_2": runner.run_experiment_2()}
            elif args.experiment == 3:
                results = {"experiment_3": runner.run_experiment_3(config_filter=args.config_filter)}
            elif args.experiment == 4:
                results = {"experiment_4": runner.run_experiment_4()}
            
            # Save results
            import json
            output_file = Path(__file__).parent / "results" / f"experiment_{args.experiment}_results.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"\nResults saved to: {output_file}")
            
        else:
            # Run all experiments
            logger.info("\nStarting all 4 experiments...")
            results = runner.run_all_experiments(save_results=True)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"ALL EXPERIMENTS COMPLETED")
        logger.info(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
        logger.info(f"{'='*80}")
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        for exp_name, exp_data in results.items():
            if exp_name == "metadata":
                continue
            
            print(f"\n{exp_name.upper().replace('_', ' ')}:")
            
            if isinstance(exp_data, dict):
                for config_name, config_data in exp_data.items():
                    if not isinstance(config_data, dict):
                        continue
                        
                    # Handle aggregated results (pooled_asr)
                    if "pooled_asr" in config_data:
                        asr = config_data["pooled_asr"]
                        fpr = config_data.get("pooled_fpr", 0.0)
                        trials = config_data.get("trials_completed", 1)
                        print(f"  {config_name:30s} ASR={asr:5.1%}  FPR={fpr:5.1%} (Trials: {trials})")
                    # Handle single trial results (legacy/backup)
                    elif "metrics" in config_data:
                        metrics = config_data["metrics"]
                        asr = metrics.get("attack_success_rate", 0)
                        fpr = metrics.get("false_positive_rate", 0)
                        print(f"  {config_name:30s} ASR={asr:5.1%}  FPR={fpr:5.1%}")
        
        print("\n" + "="*80)
        print("Results saved to:")
        print(f"  - Database: experiments.db")
        print(f"  - JSON: results/experiment_results.json")
        print(f"  - Log: experiments.log")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nExperiments interrupted by user")
        logger.info("Partial results saved to database")
        return 1
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
