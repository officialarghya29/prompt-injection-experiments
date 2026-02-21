"""Experiment runner for conducting all 4 experiments.

This module orchestrates the execution of experiments 1-4 as defined
in the experimental plan:
1. Layer Propagation (RQ1)
2. Trust Boundary Analysis (RQ2)
3. Coordinated Defense (RQ3)
4. Layer Ablation Study

Now supports repeated randomized trials for statistical rigor.
"""

import logging
import time
import json
import random
import concurrent.futures
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import RequestEnvelope, ExecutionTrace
from pipeline import DefensePipeline
from config import Config
from database import Database
from data.attack_prompts import ATTACK_PROMPTS, BENIGN_PROMPTS
from statistical_analysis import (
    wilson_score_interval,
    aggregate_trial_results,
    calculate_mcnemar_test,
    compare_configurations,
    format_asr_with_ci
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates experiments for the research paper.
    
    Experiments:
    1. Layer Propagation - Test 4 configurations (none, L2, L2-L3, full)
    2. Trust Boundary - Compare isolation modes (bad vs good)
    3. Coordinated Defense - Test isolated vs coordinated layers
    4. Layer Ablation - Remove one layer at a time
    
    Now supports repeated randomized trials for statistical rigor.
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None, 
        db_path: Optional[str] = None,
        n_trials: int = 1,
        random_seed: Optional[int] = None,
        max_workers: int = 1,
        attack_prompts: Optional[Dict[str, Any]] = None,
        benign_prompts: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration object
            db_path: Path to database
            n_trials: Number of randomized trials to run per configuration
            random_seed: Random seed for reproducibility (None = random)
            max_workers: Maximum number of parallel workers for trials
        """
        self.config = config or Config.get()
        self.db = Database(db_path or "experiments.db")
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.max_workers = max_workers
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)
            logger.info(f"Random seed set to: {self.random_seed}")
        
        # Load prompts
        self.attack_prompts = attack_prompts if attack_prompts is not None else ATTACK_PROMPTS
        self.benign_prompts = benign_prompts if benign_prompts is not None else BENIGN_PROMPTS
        
        logger.info(
            f"ExperimentRunner initialized: "
            f"{len(self.attack_prompts)} attack prompts, "
            f"{len(self.benign_prompts)} benign prompts, "
            f"{self.n_trials} trial(s) per configuration"
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'db') and self.db:
            self.db.close()
    
    def _run_configuration_with_trials(
        self,
        config_label: str,
        test_function,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single configuration multiple times with randomization.
        
        Args:
            config_label: Label for the configuration for logging
            test_function: Function that runs one trial and returns metrics
            **kwargs: Additional arguments to pass to test_function
            
        Returns:
            Aggregated results with statistics across trials
        """
        if self.n_trials == 1:
            # Single trial mode - backward compatible
            # Must still provide default orders for consistency
            return test_function(
                trial_num=1,
                attack_order=list(self.attack_prompts.keys()),
                benign_order=list(self.benign_prompts.keys()),
                **kwargs
            )
        
        # Multiple trials with randomization
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {config_label} with {self.n_trials} randomized trials (Workers: {self.max_workers})")
        logger.info(f"{'='*60}")
        
        trial_results = []
        
        def _execute_trial(trial_num):
            try:
                # Use a specific seed offset for each trial to ensure variety in parallel runs
                # but reproducibility if global seed is set
                trial_seed = self.random_seed + trial_num if self.random_seed is not None else None
                if trial_seed:
                    random.seed(trial_seed)
                
                logger.info(f"Starting Trial {trial_num}/{self.n_trials}...")
                
                # Shuffle attack order for this trial
                attack_order = list(self.attack_prompts.keys())
                random.shuffle(attack_order)
                
                benign_order = list(self.benign_prompts.keys())
                random.shuffle(benign_order)
                
                # Run this trial
                result = test_function(
                    trial_num=trial_num,
                    attack_order=attack_order,
                    benign_order=benign_order,
                    **kwargs
                )
                
                # Log completion
                metrics = result.get("metrics", {})
                logger.info(
                    f"Trial {trial_num} complete: ASR={metrics.get('attack_success_rate', 0):.1%}, "
                    f"FPR={metrics.get('false_positive_rate', 0):.1%}"
                )
                return result
            except Exception as e:
                logger.error(f"Error in Trial {trial_num}: {e}", exc_info=True)
                raise
        
        # Execute trials
        if self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_execute_trial, t): t for t in range(1, self.n_trials + 1)}
                for future in concurrent.futures.as_completed(futures):
                    trial_results.append(future.result())
        else:
            for trial_num in range(1, self.n_trials + 1):
                trial_results.append(_execute_trial(trial_num))
        
        # Sort results by trial number for consistency
        trial_results.sort(key=lambda x: x.get("trial_num", 0))
        
        # Aggregate across trials
        logger.info(f"\nAggregating {self.n_trials} trials...")
        aggregated = aggregate_trial_results(trial_results)
        
        # Add configuration metadata
        aggregated["config_name"] = config_label
        aggregated["trials_completed"] = self.n_trials
        
        # Log summary with confidence interval
        pooled_asr = aggregated["pooled_asr"]
        ci = aggregated["confidence_interval_95"]
        logger.info(
            f"{config_label} Summary: "
            f"ASR={pooled_asr:.1%} (95% CI: {ci['lower']:.1%}–{ci['upper']:.1%})"
        )
        
        return aggregated
    
    def _run_experiment_1_trial(
        self,
        trial_num: int,
        attack_order: List[str],
        benign_order: List[str],
        layer_config: Dict[str, Any],
        experiment_id: str,
        version_name: str
    ) -> Dict[str, Any]:
        """Run a single trial for Experiment 1."""
        pipeline = DefensePipeline(self.config)
        pipeline.configure_layers(**layer_config)
        
        trial_results = {
            "attack_results": [],
            "benign_results": [],
        }
        
        # Test attacks in randomized order
        for attack_id in attack_order:
            attack_data = self.attack_prompts[attack_id]
            request = RequestEnvelope(
                user_input=attack_data["text"],
                attack_label=attack_data["type"]
            )
            
            trace = pipeline.process(
                request,
                isolation_mode="good",
                experiment_id=f"{experiment_id}_{version_name}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            trial_results["attack_results"].append({
                "attack_id": attack_id,
                "attack_type": attack_data["type"],
                "attack_successful": trace.attack_successful,
                "blocked_at_layer": trace.blocked_at_layer,
                "violation_detected": trace.violation_detected,
                "latency_ms": trace.total_latency_ms,
            })
            
        # Test benign prompts in randomized order
        for benign_id in benign_order:
            benign_data = self.benign_prompts[benign_id]
            request = RequestEnvelope(
                user_input=benign_data["text"],
                attack_label=None
            )
            
            trace = pipeline.process(
                request,
                isolation_mode="good",
                experiment_id=f"{experiment_id}_{version_name}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            trial_results["benign_results"].append({
                "benign_id": benign_id,
                "false_positive": trace.violation_detected,
                "latency_ms": trace.total_latency_ms,
            })
            
        # Compute metrics for this trial
        attack_success_rate = sum(
            1 for r in trial_results["attack_results"] if r["attack_successful"]
        ) / len(trial_results["attack_results"])
        
        false_positive_rate = sum(
            1 for r in trial_results["benign_results"] if r["false_positive"]
        ) / len(trial_results["benign_results"])
        
        trial_results["metrics"] = {
            "attack_success_rate": attack_success_rate,
            "false_positive_rate": false_positive_rate,
            "total_attacks": len(trial_results["attack_results"]),
            "total_benign": len(trial_results["benign_results"]),
        }
        
        return trial_results

    def run_experiment_1(self) -> Dict[str, Any]:
        """
        Experiment 1: Layer Propagation Across Configurations (RQ1)
        
        Tests 4 configurations:
        - Version A: No defense (all layers disabled)
        - Version B: Layer 2 only (semantic analysis)
        - Version C: Layers 2-3 (semantic + context isolation)
        - Version D: Full stack (all 5 layers)
        
        Returns:
            Results dictionary with metrics per configuration
        """
        logger.info("="*80)
        logger.info("EXPERIMENT 1: Layer Propagation (RQ1)")
        logger.info("="*80)
        
        experiment_id = "exp1_layer_propagation"
        
        # Define configurations
        configs = {
            "Version_A_NoDefense": {
                "enable_layer1": False,
                "enable_layer2": False,
                "enable_layer3": False,
                "enable_layer4": True,  # LLM must stay enabled
                "enable_layer5": False,
            },
            "Version_B_Layer2Only": {
                "enable_layer1": False,
                "enable_layer2": True,
                "enable_layer3": False,
                "enable_layer4": True,
                "enable_layer5": False,
            },
            "Version_C_Layers2_3": {
                "enable_layer1": False,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": False,
            },
            "Version_D_FullStack": {
                "enable_layer1": True,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": True,
            },
        }
        
        results = {}
        
        for version_name, layer_config in configs.items():
            version_results = self._run_configuration_with_trials(
                version_name,
                self._run_experiment_1_trial,
                layer_config=layer_config,
                experiment_id=experiment_id,
                version_name=version_name
            )
            results[version_name] = version_results
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 1 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _run_experiment_2_trial(
        self,
        trial_num: int,
        attack_order: List[str],
        benign_order: List[str],
        mode: str,
        context_attacks: Dict[str, Any],
        experiment_id: str
    ) -> Dict[str, Any]:
        """Run a single trial for Experiment 2."""
        pipeline = DefensePipeline(self.config)
        pipeline.configure_layers(
            enable_layer1=True,
            enable_layer2=True,
            enable_layer3=True,
            enable_layer4=True,
            enable_layer5=True,
        )
        
        trial_results = {
            "attack_results": [],
        }
        
        # Test context override attacks in randomized order (only those present in attack_order)
        filtered_order = [aid for aid in attack_order if aid in context_attacks]
        
        for attack_id in filtered_order:
            attack_data = context_attacks[attack_id]
            request = RequestEnvelope(
                user_input=attack_data["text"],
                attack_label=attack_data["type"]
            )
            
            trace = pipeline.process(
                request,
                isolation_mode=mode,
                experiment_id=f"{experiment_id}_{mode}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            # Check if violation was due to boundary issues
            boundary_violation = any(
                "violation" in flag or "boundary" in flag
                for result in trace.layer_results
                for flag in result.flags
            )
            
            trial_results["attack_results"].append({
                "attack_id": attack_id,
                "attack_successful": trace.attack_successful,
                "boundary_violation": boundary_violation,
                "blocked_at_layer": trace.blocked_at_layer,
                "layer3_risk": next(
                    (r.risk_score for r in trace.layer_results if r.layer_name == "Layer3_Context"),
                    0.0
                ),
            })
            
        # Compute metrics for this trial
        num_attacks = len(trial_results["attack_results"])
        if num_attacks > 0:
            attack_success_rate = sum(
                1 for r in trial_results["attack_results"] if r["attack_successful"]
            ) / num_attacks
            
            boundary_violation_rate = sum(
                1 for r in trial_results["attack_results"] if r["boundary_violation"]
            ) / num_attacks
        else:
            attack_success_rate = 0.0
            boundary_violation_rate = 0.0
            
        trial_results["metrics"] = {
            "attack_success_rate": attack_success_rate,
            "false_positive_rate": 0.0, # Exp 2 doesn't focus on FP
            "boundary_violation_rate": boundary_violation_rate,
            "total_attacks": num_attacks,
        }
        
        return trial_results

    def run_experiment_2(self) -> Dict[str, Any]:
        """
        Experiment 2: Trust Boundary Violation Analysis (RQ2)
        
        Compares different context isolation modes:
        - Bad: Concatenated system + user prompt
        - Good: Role-based separation
        - Metadata: XML-tagged separation
        - Strict: Maximum isolation
        
        Focuses on context override attacks.
        
        Returns:
            Results dictionary with metrics per isolation mode
        """
        logger.info("="*80)
        logger.info("EXPERIMENT 2: Trust Boundary Analysis (RQ2)")
        logger.info("="*80)
        
        experiment_id = "exp2_trust_boundary"
        
        # Isolation modes to test
        isolation_modes = ["bad", "good", "metadata", "strict"]
        
        # Filter for context override attacks
        context_attacks = {
            aid: data for aid, data in self.attack_prompts.items()
            if "context" in data["type"].lower() or "override" in data["type"].lower()
        }
        
        logger.info(f"Testing {len(context_attacks)} context override attacks")
        
        # If no context attacks available (e.g., in quick mode), use all attacks
        if len(context_attacks) == 0:
            logger.warning(
                "No context override attacks found in prompt set. "
                "Using all attacks instead. This may happen in quick mode."
            )
            context_attacks = self.attack_prompts
        
        results = {}
        
        for mode in isolation_modes:
            mode_results = self._run_configuration_with_trials(
                mode,
                self._run_experiment_2_trial,
                mode=mode,
                context_attacks=context_attacks,
                experiment_id=experiment_id
            )
            results[mode] = mode_results
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 2 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _run_experiment_3_trial(
        self,
        trial_num: int,
        attack_order: List[str],
        benign_order: List[str],
        layer_config: Dict[str, Any],
        experiment_id: str,
        config_name: str
    ) -> Dict[str, Any]:
        """Run a single trial for Experiment 3."""
        pipeline = DefensePipeline(self.config)
        pipeline.configure_layers(**layer_config)
        
        trial_results = {
            "attack_results": [],
            "benign_results": [],
        }
        
        # Test attacks in randomized order
        for attack_id in attack_order:
            attack_data = self.attack_prompts[attack_id]
            request = RequestEnvelope(
                user_input=attack_data["text"],
                attack_label=attack_data["type"]
            )
            
            trace = pipeline.process(
                request,
                isolation_mode="good",
                experiment_id=f"{experiment_id}_{config_name}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            trial_results["attack_results"].append({
                "attack_id": attack_id,
                "attack_type": attack_data["type"],
                "attack_successful": trace.attack_successful,
                "blocked_at_layer": trace.blocked_at_layer,
            })
            
        # Test benign prompts in randomized order
        for benign_id in benign_order:
            benign_data = self.benign_prompts[benign_id]
            request = RequestEnvelope(
                user_input=benign_data["text"],
                attack_label=None
            )
            
            trace = pipeline.process(
                request,
                isolation_mode="good",
                experiment_id=f"{experiment_id}_{config_name}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            trial_results["benign_results"].append({
                "benign_id": benign_id,
                "false_positive": trace.violation_detected,
            })
            
        # Compute metrics for this trial
        attack_success_rate = sum(
            1 for r in trial_results["attack_results"] if r["attack_successful"]
        ) / len(trial_results["attack_results"])
        
        false_positive_rate = sum(
            1 for r in trial_results["benign_results"] if r["false_positive"]
        ) / len(trial_results["benign_results"])
        
        trial_results["metrics"] = {
            "attack_success_rate": attack_success_rate,
            "false_positive_rate": false_positive_rate,
            "total_attacks": len(trial_results["attack_results"]),
            "total_benign": len(trial_results["benign_results"]),
        }
        
        return trial_results

    def run_experiment_3(self, config_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Experiment 3: Coordinated vs. Isolated Layer effectiveness.
        """
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 3: COORDINATED VS. ISOLATED LAYERS")
        logger.info("="*80)
        
        experiment_id = "exp3_coordinated_defense"
        
        # Defense configurations
        configs = {
            "D1_Layer2Only": {
                "enable_layer1": False,
                "enable_layer2": True,
                "enable_layer3": False,
                "enable_layer4": True,
                "enable_layer5": False,
            },
            "D2_Layer3Only": {
                "enable_layer1": False,
                "enable_layer2": False,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": False,
            },
            "D3_Layer5Only": {
                "enable_layer1": False,
                "enable_layer2": False,
                "enable_layer3": False,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "D4_Layers2_5": {
                "enable_layer1": False,
                "enable_layer2": True,
                "enable_layer3": False,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "D5_FullWorkflow": {
                "enable_layer1": True,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": True,
            },
        }
        
        results = {}
        
        for config_name, layer_config in configs.items():
            if config_filter and config_filter not in config_name:
                continue
            config_results = self._run_configuration_with_trials(
                config_name,
                self._run_experiment_3_trial,
                layer_config=layer_config,
                experiment_id=experiment_id,
                config_name=config_name
            )
            results[config_name] = config_results
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 3 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _run_experiment_4_trial(
        self,
        trial_num: int,
        attack_order: List[str],
        benign_order: List[str],
        layer_config: Dict[str, Any],
        experiment_id: str,
        config_name: str
    ) -> Dict[str, Any]:
        """Run a single trial for Experiment 4."""
        pipeline = DefensePipeline(self.config)
        pipeline.configure_layers(**layer_config)
        
        trial_results = {
            "attack_results": [],
        }
        
        # Test attacks in randomized order
        for attack_id in attack_order:
            attack_data = self.attack_prompts[attack_id]
            request = RequestEnvelope(
                user_input=attack_data["text"],
                attack_label=attack_data["type"]
            )
            
            trace = pipeline.process(
                request,
                isolation_mode="good",
                experiment_id=f"{experiment_id}_{config_name}_t{trial_num}"
            )
            
            self.db.save_execution_trace(trace)
            
            trial_results["attack_results"].append({
                "attack_id": attack_id,
                "attack_type": attack_data["type"],
                "attack_successful": trace.attack_successful,
            })
            
        # Compute metrics for this trial
        attack_success_rate = sum(
            1 for r in trial_results["attack_results"] if r["attack_successful"]
        ) / len(trial_results["attack_results"])
        
        trial_results["metrics"] = {
            "attack_success_rate": attack_success_rate,
            "false_positive_rate": 0.0, # Exp 4 doesn't focus on FP
            "total_attacks": len(trial_results["attack_results"]),
        }
        
        return trial_results

    def run_experiment_4(self) -> Dict[str, Any]:
        """
        Experiment 4: Layer Ablation Study
        
        Systematically removes one layer at a time from the full stack
        to measure individual layer contribution.
        
        Returns:
            Results showing delta when each layer is removed
        """
        logger.info("="*80)
        logger.info("EXPERIMENT 4: Layer Ablation Study")
        logger.info("="*80)
        
        experiment_id = "exp4_layer_ablation"
        
        # Ablation configurations
        configs = {
            "Full_Stack": {
                "enable_layer1": True,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "Remove_Layer1": {
                "enable_layer1": False,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "Remove_Layer2": {
                "enable_layer1": True,
                "enable_layer2": False,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "Remove_Layer3": {
                "enable_layer1": True,
                "enable_layer2": True,
                "enable_layer3": False,
                "enable_layer4": True,
                "enable_layer5": True,
            },
            "Remove_Layer5": {
                "enable_layer1": True,
                "enable_layer2": True,
                "enable_layer3": True,
                "enable_layer4": True,
                "enable_layer5": False,
            },
        }
        
        results = {}
        
        for config_name, layer_config in configs.items():
            config_results = self._run_configuration_with_trials(
                config_name,
                self._run_experiment_4_trial,
                layer_config=layer_config,
                experiment_id=experiment_id,
                config_name=config_name
            )
            results[config_name] = config_results
        
        # Compute delta contributions
        baseline_success = results["Full_Stack"]["pooled_asr"]
        
        for config_name in ["Remove_Layer1", "Remove_Layer2", "Remove_Layer3", "Remove_Layer5"]:
            if config_name in results:
                ablated_success = results[config_name]["pooled_asr"]
                delta = ablated_success - baseline_success
                results[config_name]["contribution_delta"] = delta
                
                layer_name = config_name.replace("Remove_", "")
                logger.info(
                    f"{layer_name} contribution: Δ = {delta:+.1%} "
                    f"(removes this layer → {ablated_success:.1%} success)"
                )
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 4 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def run_all_experiments(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run all 4 experiments sequentially.
        
        Args:
            save_results: Whether to save results to JSON file
            
        Returns:
            Combined results from all experiments
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL EXPERIMENTS")
        logger.info("="*80)
        
        start_time = time.time()
        
        all_results = {
            "experiment_1": self.run_experiment_1(),
            "experiment_2": self.run_experiment_2(),
            "experiment_3": self.run_experiment_3(),
            "experiment_4": self.run_experiment_4(),
        }
        
        total_time = time.time() - start_time
        all_results["metadata"] = {
            "total_time_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ALL EXPERIMENTS COMPLETE - Total time: {total_time/60:.1f} minutes")
        logger.info(f"{'='*80}")
        
        if save_results:
            output_file = Path(__file__).parent.parent / "results" / "experiment_results.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {output_file}")
        
        return all_results


def main():
    """Main entry point for running experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = ExperimentRunner()
    results = runner.run_all_experiments(save_results=True)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Print high-level summary
    for exp_name, exp_results in results.items():
        if exp_name == "metadata":
            continue
        print(f"\n{exp_name.upper().replace('_', ' ')}:")
        
        if isinstance(exp_results, dict):
            for config_name, config_data in exp_results.items():
                if isinstance(config_data, dict) and "metrics" in config_data:
                    metrics = config_data["metrics"]
                    asr = metrics.get("attack_success_rate", 0)
                    print(f"  {config_name}: Attack Success = {asr:.1%}")


if __name__ == "__main__":
    main()
