# Chaos Engineering Suite

This module implements advanced chaos experiments for monitoring **system resilience**.

### Features
- Kubernetes chaos experiments (pod kill, network latency, CPU stress).
- System-level chaos (disk fill, CPU burn, network shaping).
- Centralized runner to execute chaos experiments programmatically.
- Docker + GitHub Actions integration for reproducibility.
- Auto-reporting for resilience verification.

### Usage

```bash
# Install dependencies
npm ci

# Run a chaos experiment
node scripts/run_experiment.js experiments/cpu_stress.json

# Generate chaos report
node scripts/report_generator.js reports/*.json
