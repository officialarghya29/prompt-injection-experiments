#!/usr/bin/env python3
"""
Create visualizations for the prompt injection defense research paper.
Updated with refinements for clarity, accuracy, and aesthetics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set global style for professional look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Curated Professional Palette
PALETTE = {
    'l1': '#FFECB3', # Light Amber
    'l2': '#BBDEFB', # Light Blue
    'l3': '#C8E6C9', # Light Green
    'l4': '#F8BBD0', # Light Pink
    'l5': '#D1C4E9', # Light Purple
    'l6': '#FFF9C4', # Light Yellow
    'adaptive': '#E0F2F1', # Teal-ish
    'border': '#455A64',   # Slate Grey
    'text': '#263238',     # Dark Charcoal
    'accent': '#1DE9B6',   # Teal Bright
    'danger': '#FF5252'    # Soft Red
}

class DrawContext:
    """Helper to manage relative geometry and consistent styling."""
    def __init__(self, ax):
        self.ax = ax
        self.arrow_props = dict(arrowstyle='-|>', lw=1.5, color=PALETTE['border'], 
                               mutation_scale=20, shrinkA=2, shrinkB=2)
        
    def draw_box(self, x, y, width, height, title, color, subtitle=None, linestyle='-', title_y_offset=0):
        rect = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                              boxstyle="round,pad=0.1", fc=color, ec=PALETTE['border'], 
                              lw=1.5, linestyle=linestyle, zorder=2)
        self.ax.add_patch(rect)
        
        # Title placement with optional offset
        self.ax.text(x, y + title_y_offset, title, ha='center', va='center', 
                     fontsize=10, fontweight='bold', color=PALETTE['text'], zorder=3)
        if subtitle:
            self.ax.text(x, y - 0.35 + title_y_offset, subtitle, ha='center', va='center', 
                         fontsize=8, style='italic', color='#546E7A', zorder=3)
        return (x, y, width, height)

    def connect(self, start_xy, end_xy, label=None, connectionstyle=None, color=None, **kwargs):
        props = self.arrow_props.copy()
        if color: props['color'] = color
        if connectionstyle: props['connectionstyle'] = connectionstyle
        props.update(kwargs)
        
        self.ax.annotate('', xy=end_xy, xytext=start_xy, arrowprops=props, zorder=1)
        if label:
            mid_x = (start_xy[0] + end_xy[0]) / 2
            mid_y = (start_xy[1] + end_xy[1]) / 2
            self.ax.text(mid_x, mid_y + 0.1, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

def create_architecture_diagram():
    """Create the architecture diagram showing the multi-layer defense system."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ctx = DrawContext(ax)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Layer Definitions
    layers = [
        ("Layer 1: Boundary Layer", PALETTE['l1'], "Normalization & Regex"),
        ("Layer 2: Semantic Layer", PALETTE['l2'], "Similarity Pattern Mapping"),
        ("Layer 3: Context Isolation", PALETTE['l3'], "Role-Separated Segments"),
        ("Layer 4: LLM Interaction", PALETTE['l4'], "Controlled Prompt Injection"),
        ("Layer 5: Output Validation", PALETTE['l5'], "Content Leakage Detection")
    ]
    
    layer_widths = 7
    layer_height = 0.6
    spacing = 1.0
    start_y = 6.8
    
    layer_centers = []
    for i, (name, color, desc) in enumerate(layers):
        y = start_y - (i * spacing)
        ctx.draw_box(6, y, layer_widths, layer_height, name, color, subtitle=desc)
        layer_centers.append((6, y))
        
        # Connect to next layer
        if i > 0:
            prev_y = layer_centers[i-1][1] - (layer_height/2)
            curr_y = y + (layer_height/2)
            ctx.connect((6, prev_y), (6, curr_y))

    # Layer 6: Feedback (Bottom)
    feedback_y = 1.2
    ctx.draw_box(6, feedback_y, 9, 0.8, "Layer 6: Feedback Coordination", 
                 PALETTE['l6'], subtitle="Continuous Evaluation & Offline Refinement", linestyle='--')
    
    # Feedback flow (Side curves)
    for i, (lx, ly) in enumerate(layer_centers):
        # From internal layers to L6
        start = (lx - layer_widths/2, ly)
        end = (lx - 3.5, feedback_y + 0.4)
        ctx.connect(start, end, connectionstyle="arc3,rad=0.3", color='#B0BEC5')

    # Adaptive Elements (Right callout)
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor=PALETTE['adaptive'], 
                      edgecolor=PALETTE['accent'], alpha=0.9, linewidth=1.5)
    ax.text(10.5, 4.0, 'Adaptive Elements\n(L2-L4, L6)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#00695C', bbox=bbox_props, zorder=4)
    
    # Connect adaptive box to L2, L3, L4
    for i in [1, 2, 3]:
        ly = layer_centers[i][1]
        ctx.connect((10.0, 4.0), (6 + layer_widths/2, ly), color=PALETTE['accent'], linestyle=':')

    # IO
    ctx.connect((1.5, 6.8), (6 - layer_widths/2, 6.8), label="User Input")
    ctx.connect((6 + layer_widths/2, 2.8), (10.5, 2.8), label="Safe Response")

    ax.text(6, 7.8, 'System-Level Workflow Model for Prompt Injection Mitigation', 
            ha='center', va='center', fontsize=15, fontweight='bold', color=PALETTE['text'])
    
    plt.savefig('visualizations_root/System-Level Workflow Model for Prompt Injection Mitigation_generated.png')
    print("Layer architecture diagram refined.")
    plt.close()

def create_asr_comparison_chart():
    """Create the attack success rate comparison chart."""
    # Precise data from paper
    configurations = ['Baseline\n(No Defense)', 'L3 Only\n(Isolation)', 'L5 Only\n(Validation)', 'Full Stack\n(Coordinated)']
    asr_values = [4.19, 37.4, 2.6, 0.00] 
    
    # 95% Confidence Intervals (Wilson Score)
    # 4.19% [2.66, 6.52] -> ~1.9 error
    # 37.4% [~36.5, ~38.3] -> ~0.9 error
    # 2.6% [~2.3, ~2.9] -> ~0.3 error
    # 0.0% [0.0, 0.71] -> ~0.7 error
    errors = [1.9, 0.9, 0.3, 0.71] 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create bar plot
    colors = ['#CB4335', '#5DADE2', '#58D68D', '#F5B041', '#AF7AC5']
    bars = ax.bar(configurations, asr_values, 
                  yerr=errors, capsize=4, 
                  color=colors, width=0.65,
                  edgecolor='#333333', linewidth=1, zorder=3)
    
    # Add labels
    for bar, value in zip(bars, asr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Styling
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax.set_title('Attack Success Rate Comparison Across Configurations\n(Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 90) # Give space for baseline
    
    # Remove top and right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('visualizations_root/asr_comparison_chart_generated.png')
    print("ASR comparison chart created: visualizations_root/asr_comparison_chart_generated.png")
    plt.close()

def create_bypass_mechanisms_viz():
    """Create the bypass mechanism analysis visualization (Pie + Bar)."""
    # Data
    mechanisms = ['Output Leakage', 'Semantic Evasion\n(Stealth)', 'Constraint\nViolations']
    counts = [1080, 1471, 126]  # Based on 9.4%, 12.8%, 1.1% of 11,490
    total_bypass = sum(counts)
    
    # Calculate percentages
    percentages = [c/total_bypass*100 for c in counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Distribution of 1,473 Bypass Mechanisms Detected Across 11,490 Traces', fontsize=14, fontweight='bold')
    
    # Pie Chart
    colors = ['#F1948A', '#85C1E9', '#F7DC6F']
    wedges, texts, autotexts = ax1.pie(counts, labels=None, 
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors, pctdistance=0.85,
                                        wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # Improve pie labels
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        ax1.annotate(mechanisms[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.2*y),
                    horizontalalignment=horizontalalignment,
                    arrowprops=dict(arrowstyle="-", color="gray"), fontsize=10, fontweight='bold')
                    
    ax1.set_title('Proportional Distribution', fontsize=12)

    # Bar Chart
    bars = ax2.bar([m.replace('\n', ' ') for m in mechanisms], counts, 
                   color=colors, edgecolor='black', linewidth=1, width=0.6)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                 f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.text(bar.get_x() + bar.get_width()/2., height - 60,
                 f'({pct:.1f}%)', ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    ax2.set_ylabel('Number of Occurrences', fontsize=11)
    ax2.set_title('Frequency Count', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('visualizations_root/bypass_mechanisms.png')
    print("Bypass mechanisms viz created: visualizations_root/bypass_mechanisms.png")
    plt.close()

def create_layer_effectiveness_flow_qualitative():
    """Create a qualitative layer effectiveness flow (No numbers)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ctx = DrawContext(ax)
    
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 11)
    ax.axis('off')
    
    layers = [
        ('Request\nBoundary', 'Sanitization', 'High Efficiency', PALETTE['l1']),
        ('Semantic\nAnalysis', 'Pattern Checks', 'Med/High Precision', PALETTE['l2']),
        ('Context\nIsolation', 'Separation', 'Systemic Safety', PALETTE['l3']),
        ('LLM\nInteraction', 'Enforcement', 'Semantic Controls', PALETTE['l4']),
        ('Output\nValidation', 'Leakage Check', 'Final Safeguard', PALETTE['l5'])
    ]
    
    # Draw blocks with uniform spacing
    x_positions = np.linspace(1.5, 9.5, 5)
    width, height = 1.6, 1.2
    
    centers = []
    for i, (x, (name, desc, qual, color)) in enumerate(zip(x_positions, layers)):
        ctx.draw_box(x, 2.8, width, height, name, color, subtitle=desc)
        ax.text(x, 1.8, qual, ha='center', va='top', fontsize=9, fontweight='bold', color='#455A64')
        centers.append((x, 2.8))
        
        # Connect to next
        if i > 0:
            prev_x = centers[i-1][0] + (width/2)
            curr_x = x - (width/2)
            ctx.connect((prev_x, 2.8), (curr_x, 2.8))
    
    # Input/Output
    ctx.connect((0.2, 2.8), (centers[0][0] - width/2, 2.8), label='User Input')
    ctx.connect((centers[-1][0] + width/2, 2.8), (10.8, 2.8), label='Safe Output')

    ax.text(5.5, 4.5, 'Qualitative Defense Effectiveness Flow', ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(5.5, 4.1, 'Cumulative risk reduction across coordinated defense layers', ha='center', va='center', fontsize=10, style='italic')
    
    plt.savefig('visualizations_root/layer_effectiveness_flow_generated.png')
    print("Layer effectiveness qualitative flow refined.")
    plt.close()

def create_statistical_significance_forest():
    """Create the statistical significance visualization (Forest Plot)."""
    # Updated to match strict paper numbers
    configs = ['Baseline', 'L3 Only', 'L5 Only', 'Full Stack']
    asr_values = [4.19, 37.4, 2.6, 0.00] 
    
    # Wilson Score Intervals
    ci_err = [1.9, 0.9, 0.3, 0.71] 
    
    # Calculate improvements relative to Baseline (4.19)
    baseline_val = 4.19
    improvements_rel = [(baseline_val - val) / baseline_val * 100 for val in asr_values[1:]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(configs))
    
    # Forest plot points
    # Increased markersize and linewidths to make tiny CIs (L5/Full Stack) more visible
    ax.errorbar(asr_values, y_pos, xerr=ci_err, fmt='o', 
                markersize=10, capsize=8, color='#2874A6', ecolor='#2874A6', 
                elinewidth=3, markeredgewidth=2)
    
    # Reference Line (Baseline)
    ax.axvline(x=4.19, color='#E74C3C', linestyle='--', alpha=0.6, label='Baseline (4.19%)')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontweight='bold', fontsize=11)
    ax.set_xlabel('Attack Success Rate (%)', fontweight='bold')
    ax.set_title('ASR Distribution with 95% Confidence Intervals', fontweight='bold', fontsize=14)
    ax.invert_yaxis() # Top-down
    
    # Add value annotations
    for i, val in enumerate(asr_values):
        if i == 0:
            # Move baseline text significantly left to ensure no overlap even with larger caps
            ax.text(val - 4.5, i, f'{val:.2f}%', va='center', ha='right', fontweight='bold')
        else:
            imp_rel = improvements_rel[i-1]
            color = '#196F3D' if imp_rel >= 0 else '#C0392B'
            sign = '↓' if imp_rel >= 0 else '↑'
            # Move labels slightly further to ensure they don't touch the error bars
            ax.text(val + 4.5, i, f'{val:.2f}% ({sign}{abs(imp_rel):.1f}%)', va='center', fontweight='bold', color=color)

    ax.set_xlim(-15, 60) # Expanded range to give text more breathing room
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('visualizations_root/statistical_significance_analysis.png')
    print("Statistical significance forest plot created: visualizations_root/statistical_significance_analysis.png")
    plt.close()

def create_attack_mitigated_flow():
    """Create the Attack Flow vs Mitigated Flow comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.2])
    
    # Vulnerable Flow
    ctx1 = DrawContext(ax1)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    ax1.text(6, 3.5, "UNPROTECTED FLOW (High Success Rate)", ha='center', va='center', 
             fontsize=12, fontweight='bold', color=PALETTE['danger'])
    
    vuln_nodes = [
        (2.0, 'Malicious Input', PALETTE['l1']),
        (5.5, 'Application Layer', PALETTE['l4']),
        (9.0, 'Direct LLM Access', PALETTE['danger'])
    ]
    
    v_width = 2.2
    v_centers = []
    for x, text, color in vuln_nodes:
        ctx1.draw_box(x, 2.0, v_width, 1.0, text, color)
        v_centers.append((x, 2.0))
        
    for i in range(len(v_centers)-1):
        ctx1.connect((v_centers[i][0] + v_width/2, 2.0), (v_centers[i+1][0] - v_width/2, 2.0), color=PALETTE['danger'])
    
    ctx1.connect((v_centers[-1][0] + v_width/2, 2.0), (11.5, 2.0), label="Attack Success", color=PALETTE['danger'])

    # Mitigated Flow
    ctx2 = DrawContext(ax2)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    ax2.text(6, 3.5, "MITIGATED FLOW (0.0% Success Rate)", ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#2E7D32')

    # Simplified layer view
    m_layers = [
        (2.0, 'Boundary', PALETTE['l1']),
        (4.5, 'Semantic / Context', PALETTE['l2']),
        (7.5, 'Safe LLM Interaction', PALETTE['l4']),
        (10.0, 'Filtered Output', PALETTE['l5'])
    ]
    m_width = 2.0
    m_centers = []
    for x, text, color in m_layers:
        ctx2.draw_box(x, 2.0, m_width, 1.0, text, color)
        m_centers.append((x, 2.0))
        
    for i in range(len(m_centers)-1):
        ctx2.connect((m_centers[i][0] + m_width/2, 2.0), (m_centers[i+1][0] - m_width/2, 2.0), color='#2E7D32')
        
    ctx2.connect((0.2, 2.0), (m_centers[0][0] - m_width/2, 2.0))
    ctx2.connect((m_centers[-1][0] + m_width/2, 2.0), (11.5, 2.0), label="Safe Outcome", color='#2E7D32')

    # L6 Context Box
    rect = mpatches.FancyBboxPatch((0.5, 0.4), 11, 2.8, boxstyle="round,pad=0.1", 
                                   facecolor=PALETTE['l6'], edgecolor=PALETTE['border'], 
                                   linewidth=1, linestyle='--', alpha=0.2, zorder=-1)
    ax2.add_patch(rect)
    ax2.text(6, 0.6, 'Layer 6: Coordination & Real-time Refinement', ha='center', va='center', 
             fontsize=10, style='italic', color='#7F8C8D')

    plt.savefig('visualizations_root/Attack Flow vs Mitigated Flow_generated.png')
    print("Attack vs Mitigated Flow diagram refined.")
    plt.close()

def create_context_isolation_diagram():
    """Create the Figure 2 diagram: Context Isolation Architecture."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ctx = DrawContext(ax)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Left Side: Input Origins
    ctx.draw_box(2, 4.5, 3, 1.0, "Trusted Instructions", PALETTE['l3'], subtitle="Fixed System Prompts")
    ctx.draw_box(2, 2.0, 3, 1.0, "User-Controlled Input", PALETTE['l1'], subtitle="Untrusted Semantic Payload")
    
    # Right Side: Isolation Layer
    # Move title to top of container to avoid overlap with internal slots
    ctx.draw_box(7, 3.25, 4, 3.5, "L3 isolation Boundary", PALETTE['l3'], 
                 subtitle="Architectural Segregation", linestyle='--', title_y_offset=1.5)
    
    # Inside L3: Slots
    ctx.draw_box(7, 4.3, 3, 0.6, "Protected Instruction Segment", '#A5D6A7')
    ctx.draw_box(7, 3.25, 3, 0.6, "Constraint Enforcement", '#FFF59D')
    ctx.draw_box(7, 2.2, 3, 0.6, "Data Interaction Segment", '#FFE082')
    
    # Connections
    ctx.connect((3.5, 4.5), (5.5, 4.3))
    ctx.connect((3.5, 2.0), (5.5, 2.2))
    
    # Central Enforcement Arrow
    ctx.connect((7, 3.9), (7, 3.6), color=PALETTE['border'])
    ctx.connect((7, 2.6), (7, 2.9), color=PALETTE['border'])
    
    # Final Output
    ctx.connect((9, 3.25), (11, 3.25), label="Safe Request Context")

    ax.text(5, 5.5, 'Context Isolation Architecture (Layer 3)', ha='center', va='center', 
             fontsize=14, fontweight='bold', color=PALETTE['text'])
    
    plt.savefig('visualizations_root/Context Isolation Architecture (Layer 3)_generated.png')
    print("Context isolation diagram refined.")
    plt.close()

def main():
    """Main function to create all visualizations."""
    viz_dir = Path('visualizations_root')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating refined visualizations...")
    create_architecture_diagram() # Figure 1
    create_context_isolation_diagram() # Figure 2
    create_attack_mitigated_flow() # Figure 3
    create_asr_comparison_chart() # Figure 4
    create_bypass_mechanisms_viz() # Figure 5
    create_statistical_significance_forest() # Figure 6
    create_layer_effectiveness_flow_qualitative() # Figure 7
    print("All visualizations created.")

if __name__ == "__main__":
    main()
