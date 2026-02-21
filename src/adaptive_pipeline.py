"""
Enhanced Pipeline with Adaptive Coordination for Experiment 6

This pipeline implements TRUE inter-layer coordination where:
- Layer 2 risk scores trigger Layer 3 isolation escalation
- Layer 4 enables enhanced monitoring based on upstream signals
- Layer 5 adjusts thresholds dynamically for high-risk requests
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time
import json
import logging

# Robust path handling
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.absolute()

# Add relevant paths to sys.path
for path in [str(current_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Standard imports
try:
    from models.request import RequestEnvelope as Request
    from models.execution_trace import ExecutionTrace
    from models.layer_result import LayerResult
    from bypass_analyzer import BypassAnalyzer
except ImportError:
    from src.models.request import RequestEnvelope as Request
    from src.models.execution_trace import ExecutionTrace
    from src.models.layer_result import LayerResult
    from src.bypass_analyzer import BypassAnalyzer

# Import layers
from layers.layer1_boundary import Layer1BoundaryValidation
from layers.layer2_semantic import Layer2SemanticAnalysis
from layers.layer3_context import Layer3ContextIsolation
from layers.layer4_llm import Layer4LLMInteraction
from layers.layer5_output import Layer5OutputValidation

logger = logging.getLogger(__name__)

class AdaptiveDefensePipeline:
    """
    Enhanced defense pipeline with TRUE adaptive coordination.
    
    Coordination Features:
    1. Per-layer risk scoring (0-1 scale)
    2. Propagation path tracking (layer-by-layer decisions)
    3. Adaptive Layer 3 isolation (escalates based on Layer 2 risk)
    4. Enhanced Layer 4 monitoring (triggered by upstream signals)
    5. Dynamic Layer 5 thresholds (adjusted for high-risk requests)
    """
    
    def __init__(
        self,
        system_prompt: str,
        layers_enabled: Optional[Dict[str, bool]] = None,
        isolation_mode: str = "good",
        coordination_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adaptive defense pipeline.
        """
        from config import Config
        self.config = Config.get()
        self.system_prompt = system_prompt
        self.layers_enabled = layers_enabled or {
            "layer1": True,
            "layer2": True,
            "layer3": True,
            "layer4": True,
            "layer5": True
        }
        self.base_isolation_mode = isolation_mode
        
        # Coordination configuration
        self.coordination_config = coordination_config or {
            "enabled": False,
            "adaptive_layer3": False,
            "adaptive_layer4": False,
            "adaptive_layer5": False,
            "layer3_escalation_thresholds": {
                "strict": 0.6,
                "metadata": 0.4
            },
            "layer4_monitoring_threshold": 0.5,
            "layer5_threshold_adjustment": 0.2
        }
        
        # Initialize layers
        self.layer1 = Layer1BoundaryValidation(self.config)
        self.layer2 = Layer2SemanticAnalysis(self.config)
        self.layer3 = Layer3ContextIsolation(self.config)
        self.layer4 = Layer4LLMInteraction(self.config)
        self.layer5 = Layer5OutputValidation(self.config)
        
        # Analysis component
        self.analyzer = BypassAnalyzer()
    
    def process_request(self, request: Request) -> ExecutionTrace:
        """
        Process request through adaptive defense pipeline.
        """
        start_time = time.time()
        
        # Initialize tracking structures
        layer_results: List[LayerResult] = []
        propagation_path: List[Dict[str, Any]] = []
        bypass_mechanisms: List[str] = []
        trust_boundary_violations: List[Dict[str, Any]] = []
        
        # Coordination context (shared across layers)
        coordination_context = {
            "upstream_risk_scores": [],
            "detected_patterns": [],
            "confidence_scores": [],
            "layer2_risk_score": None,
            "layer3_isolation_mode": self.base_isolation_mode,
            "isolation_mode_escalated": False,
            "layer4_enhanced_monitoring": False,
            "layer5_threshold_adjusted": False,
            "adaptive_decisions": []
        }
        
        blocked_at_layer = None
        final_output = None
        isolated_context = None
        
        # =================================================================
        # LAYER 1: Boundary Defense
        # =================================================================
        if self.layers_enabled.get("layer1", False):
            layer1_result = self.layer1.validate(request)
            layer_results.append(layer1_result)
            
            propagation_path.append({
                "layer": "Layer1_Boundary",
                "detection_score": getattr(layer1_result, 'risk_score', 0.0),
                "decision": "pass" if layer1_result.passed else "block",
                "reason": layer1_result.annotations.get("reason", ""),
                "flags": layer1_result.flags,
                "latency_ms": layer1_result.latency_ms
            })
            
            if not layer1_result.passed:
                blocked_at_layer = "Layer1_Boundary"
                return self._create_trace(request, layer_results, None, blocked_at_layer, time.time() - start_time, propagation_path, bypass_mechanisms, trust_boundary_violations, coordination_context)
        
        # =================================================================
        # LAYER 2: Semantic Analysis
        # =================================================================
        if self.layers_enabled.get("layer2", False):
            layer2_result = self.layer2.analyze(request)
            layer_results.append(layer2_result)
            
            layer2_risk_score = getattr(layer2_result, 'risk_score', 0.5)
            coordination_context["upstream_risk_scores"].append(layer2_risk_score)
            coordination_context["detected_patterns"].extend(layer2_result.flags)
            coordination_context["layer2_risk_score"] = layer2_risk_score
            
            propagation_path.append({
                "layer": "Layer2_Semantic",
                "detection_score": layer2_risk_score,
                "decision": "pass" if layer2_result.passed else "block",
                "reason": layer2_result.annotations.get("reason", ""),
                "flags": layer2_result.flags,
                "latency_ms": layer2_result.latency_ms
            })
            
            if not layer2_result.passed:
                blocked_at_layer = "Layer2_Semantic"
                return self._create_trace(request, layer_results, None, blocked_at_layer, time.time() - start_time, propagation_path, bypass_mechanisms, trust_boundary_violations, coordination_context)
        
        # =================================================================
        # LAYER 3: Context Isolation (ADAPTIVE)
        # =================================================================
        isolation_mode = self.base_isolation_mode
        if self.layers_enabled.get("layer3", False):
            layer2_risk = coordination_context.get("layer2_risk_score", 0.0)
            if self.coordination_config.get("adaptive_layer3") and layer2_risk is not None:
                thresholds = self.coordination_config.get("layer3_escalation_thresholds", {"strict": 0.6, "metadata": 0.4})
                if layer2_risk >= thresholds.get("strict", 0.6):
                    isolation_mode = "strict"
                    coordination_context["isolation_mode_escalated"] = True
                elif layer2_risk >= thresholds.get("metadata", 0.4):
                    isolation_mode = "metadata"
                    coordination_context["isolation_mode_escalated"] = True
            
            coordination_context["layer3_isolation_mode"] = isolation_mode
            layer3_result, isolated_context = self.layer3.isolate(
                request, 
                self.system_prompt,
                mode=isolation_mode
            )
            layer_results.append(layer3_result)
            
            propagation_path.append({
                "layer": "Layer3_Context",
                "detection_score": getattr(layer3_result, 'risk_score', 0.0),
                "decision": "pass" if layer3_result.passed else "block",
                "isolation_mode": isolation_mode,
                "reason": layer3_result.annotations.get("reason", ""),
                "flags": layer3_result.flags,
                "latency_ms": layer3_result.latency_ms
            })
            
            if not layer3_result.passed:
                blocked_at_layer = "Layer3_Context"
                return self._create_trace(request, layer_results, None, blocked_at_layer, time.time() - start_time, propagation_path, bypass_mechanisms, trust_boundary_violations, coordination_context)

        # =================================================================
        # LAYER 4: LLM Interaction (ADAPTIVE)
        # =================================================================
        if self.layers_enabled.get("layer4", False):
            enhanced_monitoring = False
            if self.coordination_config.get("adaptive_layer4"):
                max_upstream_risk = max(coordination_context["upstream_risk_scores"], default=0.0)
                threshold = self.coordination_config.get("layer4_monitoring_threshold", 0.5)
                if max_upstream_risk > threshold:
                    enhanced_monitoring = True
                    coordination_context["layer4_enhanced_monitoring"] = True
            
            # FIXED: Pass enhanced_monitoring appropriately
            if isolated_context is None:
                logger.warning(
                    f"Request {request.request_id}: Layer 3 disabled or produced no isolated context. "
                    "Layer 4 will run WITHOUT system prompt isolation — this is a reduced-security configuration."
                )
            layer4_result, llm_response = self.layer4.interact(
                request,
                isolated_context or {"messages": [{"role": "user", "content": request.user_input}]},
                enhanced_monitoring=enhanced_monitoring
            )
            layer_results.append(layer4_result)
            final_output = llm_response
            
            propagation_path.append({
                "layer": "Layer4_LLM",
                "detection_score": getattr(layer4_result, 'risk_score', 0.0),
                "decision": "pass" if layer4_result.passed else "block",
                "enhanced_monitoring": enhanced_monitoring,
                "flags": layer4_result.flags,
                "latency_ms": layer4_result.latency_ms
            })
            
            if not layer4_result.passed:
                blocked_at_layer = "Layer4_LLM"
                return self._create_trace(request, layer_results, None, blocked_at_layer, time.time() - start_time, propagation_path, bypass_mechanisms, trust_boundary_violations, coordination_context)

        # =================================================================
        # LAYER 5: Output Validation (ADAPTIVE)
        # =================================================================
        if self.layers_enabled.get("layer5", False):
            threshold_adjustment = 0.0
            if self.coordination_config.get("adaptive_layer5"):
                max_upstream_risk = max(coordination_context["upstream_risk_scores"], default=0.0)
                if max_upstream_risk > 0.6:
                    threshold_adjustment = self.coordination_config.get("layer5_threshold_adjustment", 0.2)
                    coordination_context["layer5_threshold_adjusted"] = True
            
            # FIXED: Pass threshold_adjustment appropriately
            layer5_result = self.layer5.validate(
                request, 
                final_output or "",
                threshold_adjustment=threshold_adjustment
            )
            layer_results.append(layer5_result)
            
            propagation_path.append({
                "layer": "Layer5_Output",
                "detection_score": getattr(layer5_result, 'risk_score', 0.0),
                "decision": "pass" if layer5_result.passed else "block",
                "threshold_adjusted": coordination_context["layer5_threshold_adjusted"],
                "flags": layer5_result.flags,
                "latency_ms": layer5_result.latency_ms
            })
            
            if not layer5_result.passed:
                blocked_at_layer = "Layer5_Output"

        # Final Analysis
        attack_successful = blocked_at_layer is None and request.attack_label is not None
        critical_failure = None
        if attack_successful:
             # Use static method fix if needed, but assuming analyzer exists
             critical_failure = self.analyzer.identify_critical_failure(propagation_path, True)

        total_time = time.time() - start_time
        return self._create_trace(
            request, layer_results, final_output, blocked_at_layer,
            total_time, propagation_path, bypass_mechanisms,
            trust_boundary_violations, coordination_context,
            critical_failure
        )

    def _create_trace(
        self,
        request: Request,
        layer_results: List[LayerResult],
        final_output: Optional[str],
        blocked_at_layer: Optional[str],
        total_time: float,
        propagation_path: List[Dict[str, Any]],
        bypass_mechanisms: List[str],
        trust_boundary_violations: List[Dict[str, Any]],
        coordination_context: Dict[str, Any],
        critical_failure: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Create execution trace with all coordination metrics."""
        
        return ExecutionTrace(
            request_id=request.request_id,
            session_id=request.session_id,
            user_input=request.user_input,
            attack_label=request.attack_label,
            attack_successful=(blocked_at_layer is None and request.attack_label is not None),
            violation_detected=blocked_at_layer is not None,
            blocked_at_layer=blocked_at_layer,
            final_output=final_output,
            total_latency_ms=total_time * 1000,
            configuration={
                "layers_enabled": self.layers_enabled,
                "isolation_mode": self.base_isolation_mode,
                "coordination": self.coordination_config
            },
            timestamp=datetime.now(),
            layer_results=layer_results,
            coordination_enabled=self.coordination_config.get("enabled", False),
            coordination_context=coordination_context,
            propagation_path=propagation_path,
            bypass_mechanisms=bypass_mechanisms,
            trust_boundary_violations=trust_boundary_violations,
            critical_failure_point=critical_failure
        )
