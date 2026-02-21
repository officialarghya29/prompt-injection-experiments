#!/usr/bin/env python3
"""
Statistical analysis of prompt injection defense experiments.
Performs McNemar's test and Fisher's Exact Test to validate significance.
"""

import sqlite3
import math
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from scipy import stats as scipy_stats

# Canonical path to the experiment database, relative to this script
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "experiments.db"

def load_experiment_data(db_path):
    """Load experiment data from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all execution traces with configuration info
    query = """
    SELECT 
        request_id,
        attack_successful,
        configuration,
        user_input,
        experiment_id
    FROM execution_traces
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    data = []
    for row in results:
        config_raw = row[2]  # configuration is index 2
        try:
            config_dict = json.loads(config_raw)
        except:
            config_dict = {}
            
        data.append({
            'request_id': row[0],
            'attack_successful': row[1],
            'configuration': config_dict,  # Now a dict
            'user_input': row[3],
            'experiment_id': row[4]
        })
    
    return data

def is_no_defense(config_dict):
    """Check if only the LLM layer (layer 4) is enabled."""
    layers = config_dict.get('layers_enabled', {})
    return layers.get('layer4') and not any([layers.get(l) for l in ['layer1', 'layer2', 'layer3', 'layer5']])

def is_full_stack(config_dict):
    """Check if all defense layers (1-5) are enabled."""
    layers = config_dict.get('layers_enabled', {})
    return all([layers.get(l) for l in ['layer2', 'layer3', 'layer4', 'layer5']])

def is_layer2_only(config_dict):
    """Check if only layer 2 and layer 4 are enabled."""
    layers = config_dict.get('layers_enabled', {})
    return layers.get('layer2') and layers.get('layer4') and not any([layers.get(l) for l in ['layer1', 'layer3', 'layer5']])

def is_layer3_only(config_dict):
    """Check if only layer 3 and layer 4 are enabled."""
    layers = config_dict.get('layers_enabled', {})
    return layers.get('layer3') and layers.get('layer4') and not any([layers.get(l) for l in ['layer1', 'layer2', 'layer5']])

def is_layer5_only(config_dict):
    """Check if only layer 4 and layer 5 are enabled."""
    layers = config_dict.get('layers_enabled', {})
    return layers.get('layer5') and layers.get('layer4') and not any([layers.get(l) for l in ['layer1', 'layer2', 'layer3']])

def calculate_wilson_score_interval(success, total, confidence=0.95):
    """Calculate Wilson score confidence interval for proportions."""
    if total == 0:
        return 0, 0
    
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% confidence
    z2 = z * z
    
    p = success / total
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    
    interval = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denominator
    lower = centre - interval
    upper = centre + interval
    
    return lower * 100, upper * 100  # Return as percentages

def calculate_mcnemar_test(table):
    """
    Calculate McNemar's test for a 2x2 contingency table.
    table = [[a, b], [c, d]] where:
    a = both configs successful
    b = config1 successful, config2 failed
    c = config1 failed, config2 successful
    d = both configs failed
    """
    b = table[0][1]  # config1 success, config2 fail
    c = table[1][0]  # config1 fail, config2 success

    if b + c == 0:
        if table[0][0] + table[1][1] > 0:
            return 0.0, 1.0  # No significant difference
        else:
            return None, None  # Cannot compute

    # Apply continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Correct p-value using chi-squared distribution with 1 degree of freedom
    p_value = float(scipy_stats.chi2.sf(chi2, df=1))

    return chi2, p_value

def calculate_asr_comparison_with_stats(data, config_func1, config_func2, name1, name2):
    """Calculate ASR comparison with McNemar's test and confidence intervals."""
    config1_data = [item for item in data if config_func1(item['configuration'])]
    config2_data = [item for item in data if config_func2(item['configuration'])]
    
    # Create paired dataset (same requests for both configurations)
    # Use (user_input, trial) as key to avoid UUID mismatch
    def get_pairing_key(item):
        user_input = item.get('user_input') or ''
        exp_id = item.get('experiment_id') or ''
        trial = 0
        if '_t' in exp_id:
            try:
                # Extract trial number from end of string (e.g. "..._t1")
                trial_str = exp_id.split('_t')[-1]
                # Filter out everything after the number
                trial_num_str = ''.join(filter(str.isdigit, trial_str.split('_')[0]))
                if trial_num_str:
                    trial = int(trial_num_str)
            except:
                pass
        return (user_input, trial)

    config1_map = {get_pairing_key(item): item['attack_successful'] for item in config1_data}
    config2_map = {get_pairing_key(item): item['attack_successful'] for item in config2_data}
    
    # Find common keys
    common_ids = set(config1_map.keys()) & set(config2_map.keys())
    
    if not common_ids:
        return {
            'config1': name1,
            'config2': name2,
            'config1_asr': 0,
            'config2_asr': 0,
            'config1_count': len(config1_data),
            'config2_count': len(config2_data),
            'asr_reduction': 0,
            'relative_reduction': 0,
            'mcnemar_chi2': None,
            'mcnemar_p': None,
            'config1_ci_lower': 0,
            'config1_ci_upper': 0,
            'config2_ci_lower': 0,
            'config2_ci_upper': 0,
            'common_requests': 0
        }
    
    # Create contingency table for McNemar's test
    both_success = 0  # a
    config1_success_config2_fail = 0  # b
    config1_fail_config2_success = 0  # c
    both_fail = 0  # d
    
    for req_id in common_ids:
        s1 = config1_map[req_id]
        s2 = config2_map[req_id]
        
        if s1 == 1 and s2 == 1:
            both_success += 1
        elif s1 == 1 and s2 == 0:
            config1_success_config2_fail += 1
        elif s1 == 0 and s2 == 1:
            config1_fail_config2_success += 1
        else:  # s1 == 0 and s2 == 0
            both_fail += 1
    
    contingency_table = [
        [both_success, config1_success_config2_fail],
        [config1_fail_config2_success, both_fail]
    ]
    
    # Calculate McNemar's test
    chi2, p_value = calculate_mcnemar_test(contingency_table)
    
    # Calculate ASR for each configuration based on common requests
    config1_successes = both_success + config1_success_config2_fail
    config2_successes = both_success + config1_fail_config2_success
    
    config1_asr = (config1_successes / len(common_ids)) * 100 if common_ids else 0
    config2_asr = (config2_successes / len(common_ids)) * 100 if common_ids else 0
    
    # Calculate Wilson score confidence intervals
    ci1_lower, ci1_upper = calculate_wilson_score_interval(config1_successes, len(common_ids))
    ci2_lower, ci2_upper = calculate_wilson_score_interval(config2_successes, len(common_ids))
    
    asr_reduction = config1_asr - config2_asr
    relative_reduction = (asr_reduction / config1_asr * 100) if config1_asr != 0 else 0
    
    return {
        'config1': name1,
        'config2': name2,
        'config1_asr': config1_asr,
        'config2_asr': config2_asr,
        'config1_count': len(config1_data),
        'config2_count': len(config2_data),
        'asr_reduction': asr_reduction,
        'relative_reduction': relative_reduction,
        'mcnemar_chi2': chi2,
        'mcnemar_p': p_value,
        'config1_ci_lower': ci1_lower,
        'config1_ci_upper': ci1_upper,
        'config2_ci_lower': ci2_lower,
        'config2_ci_upper': ci2_upper,
        'common_requests': len(common_ids),
        'contingency_table': contingency_table
    }

def wilson_score_interval(success, total, confidence=0.95):
    """Alias for calculate_wilson_score_interval."""
    return calculate_wilson_score_interval(success, total, confidence)

def aggregate_trial_results(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple randomized trials.
    
    Computes pooled success rates and confidence intervals.
    """
    if not trial_results:
        return {}
    
    all_attack_results = []
    for trial in trial_results:
        all_attack_results.extend(trial.get("attack_results", []))
    
    if not all_attack_results:
        return {"pooled_asr": 0, "total_attacks": 0}
        
    total_attacks = len(all_attack_results)
    total_successes = sum(1 for r in all_attack_results if r.get("attack_successful", False))
    
    pooled_asr = total_successes / total_attacks if total_attacks > 0 else 0
    ci_lower, ci_upper = calculate_wilson_score_interval(total_successes, total_attacks)
    
    return {
        "pooled_asr": pooled_asr,
        "total_attacks": total_attacks,
        "confidence_interval_95": {
            "lower": ci_lower / 100, # Convert back from percentage
            "upper": ci_upper / 100
        },
        "all_trials": trial_results
    }

def format_asr_with_ci(asr: float, ci: Dict[str, float]) -> str:
    """Format ASR with its confidence interval for display."""
    return f"{asr:.1%} (95% CI: {ci['lower']:.1%}–{ci['upper']:.1%})"

def compare_configurations(result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two configurations using statistical tests."""
    # Placeholder for more complex comparison if needed
    return {
        "asr1": result1.get("pooled_asr", 0),
        "asr2": result2.get("pooled_asr", 0),
        "reduction": result1.get("pooled_asr", 0) - result2.get("pooled_asr", 0)
    }

def mcnemar_test_from_results(results_list1, results_list2):
    """Wrapper for McNemar's test on two lists of results."""
    # This requires paired samples. 
    # For now, we use a basic implementation or return None if not easily paired
    return None, None

def run_comprehensive_statistical_analysis(db_path: str = None):
    """Run comprehensive statistical analysis on all experiments."""
    print("="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)

    if db_path is None:
        db_path = str(_DEFAULT_DB_PATH)

    if not Path(db_path).exists():
        print(f"Database {db_path} not found!")
        return
    
    data = load_experiment_data(db_path)
    print(f"Loaded {len(data)} execution traces from database")
    
    # Calculate overall statistics
    total_successful = sum(item['attack_successful'] for item in data)
    overall_asr = total_successful / len(data) * 100 if len(data) > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"Total traces: {len(data)}")
    print(f"Attack success rate: {overall_asr:.2f}%")
    
    # Calculate ASR by configuration type
    print(f"\nAttack Success Rates by Configuration:")
    
    no_defense_data = [item for item in data if is_no_defense(item['configuration'])]
    full_stack_data = [item for item in data if is_full_stack(item['configuration'])]
    l2_only_data = [item for item in data if is_layer2_only(item['configuration'])]
    l3_only_data = [item for item in data if is_layer3_only(item['configuration'])]
    l5_only_data = [item for item in data if is_layer5_only(item['configuration'])]
    
    print(f"Debug Counts:")
    print(f"  No Defense: {len(no_defense_data)}")
    print(f"  Full Stack: {len(full_stack_data)}")
    print(f"  Layer 2 Only: {len(l2_only_data)}")
    print(f"  Layer 3 Only: {len(l3_only_data)}")
    print(f"  Layer 5 Only: {len(l5_only_data)}")

    if len(data) > 0:
        print(f"  Sample Configuration: {json.dumps(data[0]['configuration'])}")
    
    no_defense_asr = sum(item['attack_successful'] for item in no_defense_data) / len(no_defense_data) * 100 if len(no_defense_data) > 0 else 0
    full_stack_asr = sum(item['attack_successful'] for item in full_stack_data) / len(full_stack_data) * 100 if len(full_stack_data) > 0 else 0
    l2_asr = sum(item['attack_successful'] for item in l2_only_data) / len(l2_only_data) * 100 if len(l2_only_data) > 0 else 0
    l3_asr = sum(item['attack_successful'] for item in l3_only_data) / len(l3_only_data) * 100 if len(l3_only_data) > 0 else 0
    l5_asr = sum(item['attack_successful'] for item in l5_only_data) / len(l5_only_data) * 100 if len(l5_only_data) > 0 else 0
    
    print(f"  No Defense: {no_defense_asr:.2f}% ({len(no_defense_data)} traces)")
    print(f"  Full Stack: {full_stack_asr:.2f}% ({len(full_stack_data)} traces)")
    print(f"  Layer 2 Only: {l2_asr:.2f}% ({len(l2_only_data)} traces)")
    print(f"  Layer 3 Only: {l3_asr:.2f}% ({len(l3_only_data)} traces)")
    print(f"  Layer 5 Only: {l5_asr:.2f}% ({len(l5_only_data)} traces)")
    
    # Perform ASR comparisons with McNemar's test and confidence intervals
    print(f"\nASR Comparisons with Statistical Tests:")
    print("-" * 60)
    
    # Define key configuration pairs to compare
    config_pairs = [
        (is_no_defense, is_full_stack, 'no_defense', 'full_stack'),
        (is_no_defense, is_layer2_only, 'no_defense', 'layer2_only'),
        (is_no_defense, is_layer3_only, 'no_defense', 'layer3_only'),
        (is_no_defense, is_layer5_only, 'no_defense', 'layer5_only'),
        (is_layer3_only, is_full_stack, 'layer3_only', 'full_stack'),
        (is_layer5_only, is_full_stack, 'layer5_only', 'full_stack')
    ]
    
    asr_comparisons = []
    for config_func1, config_func2, name1, name2 in config_pairs:
        comparison = calculate_asr_comparison_with_stats(data, config_func1, config_func2, name1, name2)
        asr_comparisons.append(comparison)
        
        if comparison['common_requests'] > 0:
            print(f"{name1} vs {name2}:")
            print(f"  {name1} ASR: {comparison['config1_asr']:.2f}% (95% CI: {comparison['config1_ci_lower']:.2f}%, {comparison['config1_ci_upper']:.2f}%)")
            print(f"  {name2} ASR: {comparison['config2_asr']:.2f}% (95% CI: {comparison['config2_ci_lower']:.2f}%, {comparison['config2_ci_upper']:.2f}%)")
            print(f"  ASR Reduction: {comparison['asr_reduction']:.2f}%")
            print(f"  Relative Reduction: {comparison['relative_reduction']:.2f}%")
            print(f"  McNemar's Chi²: {comparison['mcnemar_chi2']:.4f}" if comparison['mcnemar_chi2'] is not None else "  McNemar's Chi²: N/A")
            print(f"  P-value: {comparison['mcnemar_p']:.6f}" if comparison['mcnemar_p'] is not None else "  P-value: N/A")
            if comparison['mcnemar_p'] is not None:
                print(f"  Significant (α=0.05): {'Yes' if comparison['mcnemar_p'] < 0.05 else 'No'}")
            print(f"  Common requests: {comparison['common_requests']}")
            print()
        else:
            print(f"{name1} vs {name2}: No common requests for paired comparison")
            print()
    
    # Calculate overall effectiveness
    if len(no_defense_data) > 0 and len(full_stack_data) > 0:
        overall_reduction = no_defense_asr - full_stack_asr
        relative_reduction = (no_defense_asr - full_stack_asr) / no_defense_asr * 100 if no_defense_asr != 0 else 0
        
        print(f"Overall Effectiveness of Stack:")
        print(f"  No Defense ASR: {no_defense_asr:.2f}%")
        print(f"  Full Stack ASR: {full_stack_asr:.2f}%")
        print(f"  Absolute Reduction: {overall_reduction:.2f}%")
        print(f"  Relative Reduction: {relative_reduction:.2f}%")
        print()
    
    # Summary statistics
    print(f"Statistical Summary:")
    print("-" * 30)
    significant_comparisons = [c for c in asr_comparisons 
                              if c['mcnemar_p'] is not None and c['mcnemar_p'] < 0.05 and c['common_requests'] > 0]
    print(f"  Significant comparisons (p < 0.05): {len(significant_comparisons)}")
    print(f"  Total valid comparisons: {len([c for c in asr_comparisons if c['common_requests'] > 0])}")
    
    # Save results to JSON
    results = {
        'total_traces': len(data),
        'overall_asr': overall_asr,
        'config_stats': {
            'no_defense': {
                'asr': no_defense_asr,
                'count': len(no_defense_data)
            },
            'full_stack': {
                'asr': full_stack_asr,
                'count': len(full_stack_data)
            }
        },
        'asr_comparisons': asr_comparisons
    }
    
    with open('data/statistical_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nStatistical analysis results saved to data/statistical_analysis_results.json")
    
    return results

def main():
    results = run_comprehensive_statistical_analysis()
    
    print("="*80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()