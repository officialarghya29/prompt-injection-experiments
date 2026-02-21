#!/usr/bin/env python3
"""
Experiment 6: Adaptive Coordination Test Runner

Runs one of 4 configurations:
1. isolated: No coordination (baseline)
2. adaptive_l3: Layer 3 adapts based on Layer 2
3. adaptive_l4: Layer 4 enhanced monitoring
4. full_adaptive: All adaptive features enabled
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from adaptive_pipeline import AdaptiveDefensePipeline
from database import Database
from config import SYSTEM_PROMPT
from models.request import RequestEnvelope as Request

# Load attack prompts from data/ directory in root
try:
    from data.attack_prompts import ATTACK_PROMPTS
except ImportError:
    # Fallback for different working directories
    import imp
    try:
        attack_prompts_path = project_root / "data" / "attack_prompts.py"
        attack_prompts_mod = imp.load_source('attack_prompts', str(attack_prompts_path))
        ATTACK_PROMPTS = attack_prompts_mod.ATTACK_PROMPTS
    except Exception as e:
        print(f"❌ Error loading attack prompts: {e}")
        ATTACK_PROMPTS = {}


# Configuration presets for each pod
COORDINATION_CONFIGS = {
    "isolated": {
        "enabled": False,
        "adaptive_layer3": False,
        "adaptive_layer4": False,
        "adaptive_layer5": False
    },
    "adaptive_l3": {
        "enabled": True,
        "adaptive_layer3": True,
        "adaptive_layer4": False,
        "adaptive_layer5": False,
        "layer3_escalation_thresholds": {
            "strict": 0.6,
            "metadata": 0.4
        }
    },
    "adaptive_l4": {
        "enabled": True,
        "adaptive_layer3": False,
        "adaptive_layer4": True,
        "adaptive_layer5": False,
        "layer4_monitoring_threshold": 0.5
    },
    "full_adaptive": {
        "enabled": True,
        "adaptive_layer3": True,
        "adaptive_layer4": True,
        "adaptive_layer5": True,
        "layer3_escalation_thresholds": {
            "strict": 0.6,
            "metadata": 0.4
        },
        "layer4_monitoring_threshold": 0.5,
        "layer5_threshold_adjustment": 0.2
    }
}


def run_experiment(config_name: str, output_dir: str, trials: int = 5):
    """
    Run experiment with specified configuration.
    
    Args:
        config_name: One of: isolated, adaptive_l3, adaptive_l4, full_adaptive
        output_dir: Directory to save results
        trials: Number of trials per attack
    """
    print(f"🚀 Starting Experiment 6: {config_name.upper()} Configuration")
    print(f"⏱️  Start time: {datetime.now().isoformat()}")
    print()
    
    # Get coordination configuration
    coord_config = COORDINATION_CONFIGS[config_name]
    print(f"📋 Coordination Settings:")
    print(f"   Enabled: {coord_config['enabled']}")
    print(f"   Adaptive Layer 3: {coord_config.get('adaptive_layer3', False)}")
    print(f"   Adaptive Layer 4: {coord_config.get('adaptive_layer4', False)}")
    print(f"   Adaptive Layer 5: {coord_config.get('adaptive_layer5', False)}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database and run experiments
    db_path = os.path.join(output_dir, f"exp6_{config_name}.db")
    with Database(db_path) as db:
        print(f"📊 Database: {db_path}")
        print()
        
        # Initialize pipeline
        pipeline = AdaptiveDefensePipeline(
            system_prompt=SYSTEM_PROMPT,
            layers_enabled={
                "layer1": True,
                "layer2": True,
                "layer3": True,
                "layer4": True,
                "layer5": True
            },
            isolation_mode="good",
            coordination_config=coord_config
        )
        
        # Run experiments
        total_traces = 0
        successful_attacks = 0
        blocked_attacks = 0
        errors = 0
        
        adaptive_events = {
            "layer3_escalations": 0,
            "layer4_enhanced_monitoring": 0,
            "layer5_threshold_adjustments": 0
        }
        
        start_time = time.time()
        
        print(f"🔬 Running {len(ATTACK_PROMPTS)} attacks × {trials} trials = {len(ATTACK_PROMPTS) * trials} traces")
        print()

        for attack_id, attack_data in ATTACK_PROMPTS.items():
            attack_prompt = attack_data["text"]
            attack_type = attack_data["type"]
            
            print(f"Testing {attack_id} ({attack_type})...", end=" ", flush=True)
            
            for trial in range(trials):
                try:
                    # Create request
                    request = Request(
                        user_input=attack_prompt,
                        session_id="default",
                        metadata={
                            "attack_id": attack_id,
                            "attack_type": attack_type,
                            "trial": trial + 1,
                            "experiment": "exp6_coordination",
                            "config": config_name
                        }
                    )
                    
                    # Process through pipeline
                    trace = pipeline.process_request(request)
                    
                    # Track adaptive events
                    if trace.coordination_context.get("isolation_mode_escalated"):
                        adaptive_events["layer3_escalations"] += 1
                    if trace.coordination_context.get("layer4_enhanced_monitoring"):
                        adaptive_events["layer4_enhanced_monitoring"] += 1
                    if trace.coordination_context.get("layer5_threshold_adjusted"):
                        adaptive_events["layer5_threshold_adjustments"] += 1
                    
                    # Save to database
                    db.save_execution_trace(trace, user_input=attack_prompt)
                    
                    # Update counters
                    total_traces += 1
                    if trace.attack_successful:
                        successful_attacks += 1
                    else:
                        blocked_attacks += 1
                    
                except Exception as e:
                    print(f"\n❌ Error on {attack_id} trial {trial + 1}: {e}")
                    errors += 1
            
            print(f"✓ ({successful_attacks}/{total_traces} succeeded so far)")
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        asr = (successful_attacks / total_traces * 100) if total_traces > 0 else 0
        
        # Generate summary
        summary = {
            "experiment": "exp6_coordination",
            "config": config_name,
            "coordination_enabled": coord_config["enabled"],
            "coordination_settings": coord_config,
            "total_traces": total_traces,
            "successful_attacks": successful_attacks,
            "blocked_attacks": blocked_attacks,
            "errors": errors,
            "attack_success_rate": asr,
            "elapsed_time_seconds": elapsed_time,
            "adaptive_events": adaptive_events,
            "database": db_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, f"exp6_{config_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print()
        print("=" * 80)
        print(f"✅ EXPERIMENT COMPLETE: {config_name.upper()}")
        print("=" * 80)
        print(f"Total traces: {total_traces}")
        print(f"Successful attacks: {successful_attacks}")
        print(f"Blocked attacks: {blocked_attacks}")
        print(f"Attack Success Rate: {asr:.2f}%")
        print(f"Errors: {errors}")
        print()
        
        if coord_config["enabled"]:
            print("🔄 Adaptive Coordination Events:")
            print(f"   Layer 3 escalations: {adaptive_events['layer3_escalations']}")
            print(f"   Layer 4 enhanced monitoring: {adaptive_events['layer4_enhanced_monitoring']}")
            print(f"   Layer 5 threshold adjustments: {adaptive_events['layer5_threshold_adjustments']}")
            print()
        
        print(f"⏱️  Elapsed time: {elapsed_time:.1f}s")
        print(f"📊 Database: {db_path}")
        print(f"📄 Summary: {summary_path}")
        print()
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 6 with adaptive coordination"
    )
    parser.add_argument(
        "--config",
        choices=["isolated", "adaptive_l3", "adaptive_l4", "full_adaptive"],
        required=True,
        help="Configuration to test"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per attack (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        summary = run_experiment(
            config_name=args.config,
            output_dir=args.output,
            trials=args.trials
        )
        
        print("✅ Experiment completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
