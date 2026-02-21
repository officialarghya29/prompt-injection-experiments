"""
Bypass Analysis Engine
Identifies common patterns in injection attempts that bypass defenses.
"""

from typing import List, Dict, Any, Optional
import json
import logging

from models.layer_result import LayerResult
from models.request import RequestEnvelope

logger = logging.getLogger(__name__)


class BypassAnalyzer:
    """Analyzes execution traces to identify defense bypass patterns."""

    def __init__(self):
        self.common_bypass_patterns = [
            "instruction_override",
            "character_encoding",
            "few_shot_distraction",
            "jailbreak_prefix",
            "context_override",
            "semantic_polite",
        ]

    def analyze_failure(
        self,
        request: RequestEnvelope,
        layer_results: List[LayerResult],
    ) -> Dict[str, Any]:
        """Analyze why an attack successfully bypassed layers."""
        passed_layers = [r.layer_name for r in layer_results if r.passed]

        # Heuristic pattern detection
        patterns_detected = []
        user_input = request.user_input.lower()

        if "ignore all previous instructions" in user_input or "disregard" in user_input:
            patterns_detected.append("instruction_override")
        if "\\u" in user_input or "0x" in user_input or "base64" in user_input:
            patterns_detected.append("character_encoding")
        if any(kw in user_input for kw in ["pretend", "imagine", "roleplay", "dan", "simulation"]):
            patterns_detected.append("jailbreak_prefix")
        if any(kw in user_input for kw in ["[system]", "admin_override", "debug mode", "maintenance"]):
            patterns_detected.append("context_override")
        if any(kw in user_input for kw in ["security researcher", "debugging", "educational", "audit"]):
            patterns_detected.append("semantic_polite")

        return {
            "request_id": request.request_id,
            "bypassed_layers": passed_layers,
            "patterns_detected": patterns_detected,
            "input_length": len(user_input),
        }

    def identify_critical_failure(
        self,
        propagation_path: List[Dict[str, Any]],
        attack_succeeded: bool,
    ) -> Optional[str]:
        """
        Identify which layer was the critical failure point in a successful attack.

        Returns the layer name where the attack most critically slipped through,
        or None if no critical failure is identified (or attack did not succeed).
        """
        if not attack_succeeded or not propagation_path:
            return None

        # Find the layer with the highest risk score that still passed
        critical_layer = None
        max_risk = 0.0

        for step in propagation_path:
            layer_name = step.get("layer", "")
            decision = step.get("decision", "pass")
            risk_score = step.get("detection_score", 0.0)

            if decision == "pass" and risk_score > max_risk:
                max_risk = risk_score
                critical_layer = layer_name

        if critical_layer:
            logger.debug(
                f"Critical failure point: {critical_layer} (risk_score={max_risk:.3f}, "
                f"attack passed through)"
            )

        return critical_layer
