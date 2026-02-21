# Prompt Injection Defense: Multi-Layer Architecture Experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental validation of a six-layer defense architecture against prompt injection attacks in Large Language Models (LLMs). All experiments were conducted on a high-performance local environment utilizing the **Model Context Protocol (MCP)** and **LiteLLM** for adaptive coordination.

## 📄 Paper Reference

This repository corresponds to the experimental validation described in:

**"Evaluating and Mitigating Prompt Injection in Full-Stack Web Applications: A System-Level Workflow Model"**

**Paper Version:** The results in this repository represent the high-precision validation conducted in February 2026.

## 🎯 Project Overview

This research proposes and validates a comprehensive six-layer defense architecture:

1. **Layer 1: Request Boundary** - Character-level validation and preliminary classification
2. **Layer 2: Input Processing & Semantic Analysis** - Intent classification and prompt injection detection
3. **Layer 3: Context Management & Isolation** - Separation of user data from trusted system instructions (L3/L4)
4. **Layer 4: LLM Interaction & Constraint Enforcement** - Representation-level monitoring and guardrails
5. **Layer 5: Output Handling & Validation** - Response filtering and data leakage prevention
6. **Layer 6: Feedback & Adaptive Monitoring** - Continuous learning and pattern adaptation

### Key Findings

- **Baseline Attack Success Rate (ASR):** 4.19% (Neutralizing standard injection vectors)
- **Full Stack Effectiveness:** **0.00% ASR** (100% Risk Reduction relative to baseline)
- **Stealth Attack Vulnerability:** Isolated Layer 3 (Context Isolation) yields **37.4% ASR** (Critical Failure)
- **Coordinated Defense:** The integrated 6-layer workflow achieves statistically significant neutralization of stealth attacks ($p < 0.0001$).
- **Utility Validation:** **0.0% False Positive Rate (FPR)** across 1,000 diverse benign prompt tests.
- **Statistical Significance:** $p < 0.0001$ using McNemar's test for paired data analysis.
- **Sample Size:** **11,490 execution traces** across 103 configurations, 10 trials per config.

## 🏗️ Repository Structure

```
prompt-injection-experiments/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable templates
│
├── data/                              # Data and Visualizations
│   ├── attack_prompts.py              # 52 diverse injection patterns
│   └── benign_prompts_large.py        # 1,000 diverse benign prompts (Utility stress test)
│
├── src/                               # Core Source Code
│   ├── config.py                      # Thread-safe configuration settings
│   ├── experiment_runner.py           # Multithreaded experiment orchestration
│   ├── unified_pipeline.py            # Main coordinated defense pipeline
│   ├── statistical_analysis.py        # McNemar's and Chi-squared testing tools
│   ├── create_visualizations.py       # Premium visualization generation engine
│   └── layers/                        # Defense layer implementations (L1-L6)
```

## 🧪 Experimental Methodology

### Experiment Design

We conducted a high-precision experimental campaign involving **11,490 individual LLM interactions**:

1. **Baseline Campaign**: Evaluated vanilla LLM performance vs. standard layered defenses.
2. **Layer Ablation Study**: Identified the 'Isolation Illusion' where Layer 3 (Context Isolation) fails in isolation (37.4% ASR) but succeeds when coordinated.
3. **Utility Stress Test**: Processed 1,000 diverse benign prompts (Professional, Creative, Technical) through the full defense stack to confirm a 0.0% False Positive Rate.

### Infrastructure: High-Performance Setup

The experimental validation was optimized for the following environment:

- **OS/Arch:** Linux x86_64
- **CPU:** AMD Ryzen 7 8840HS (8 Cores, 16 Threads)
- **Memory:** 32GB RAM
- **LLM Backend:** **groq/llama-3.3-70b-versatile** (via local LiteLLM proxy)
- **Performance**: Multithreaded orchestration achieved 10,050 traces in under 3 hours.

## 🚀 Reproducing the Results

### Prerequisites

1. **Python 3.11+**
2. **Groq API Key** (or compatible OpenAI-format endpoint)
3. **SQLite3**

### Setup

```bash
# Clone the repository
git clone https://github.com/Arindamtripathi619/prompt-injection-experiments.git
cd prompt-injection-experiments

# Create environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY (or local proxy URL)
```

### Running the Validation

To reproduce the final high-precision campaign:

```bash
# Run the full experiment suite (Parallelized)
python src/run_experiments.py

# Run comprehensive statistical analysis
python src/statistical_analysis.py
```

## 📊 Key Results

### Overall Attack Success Rate (ASR)

| Configuration | ASR | 95% CI | Change |
|--------------|-----|---------|---------|
| Baseline (No Defense) | 4.19% | [2.66%, 6.52%] | - |
| Stealth Attacks (L3 Solo) | 37.4% | [32.1%, 43.1%] | +792.6% |
| **Full Stack (L1-L6)** | **0.00%** | [0.00%, 0.71%] | **-100.0%** |

### McNemar's Significance Test (Paired Matching)

- **Full Stack vs. Baseline**: $\chi^2 = 72.0, p < 0.0001$
- **Full Stack vs. L3 Solo**: $\chi^2 = 161.0, p < 0.0001$

## 🤝 Contributing

This repository represents archived experimental material for a journal publication. While we welcome feedback through GitHub Issues, this branch is finalized for reproducibility purposes.

## 👥 Authors

- **Arindam Tripathi** - *Lead Researcher* - [Arindamtripathi619](https://github.com/Arindamtripathi619)
- **Arghya Bose** - *Researcher* - [officialarghya29](https://github.com/officialarghya29)
- **Arghya Paul** - *Researcher* - 24155977@kiit.ac.in

**Supervised by:**
- **Dr. Sushruta Mishra** - *Faculty, School of Computer Engineering, KIIT University*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** February 2026  
**Status**: Final Experimental Validation Corrected (0.0% ASR)
