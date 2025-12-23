# Ultrasound RL Navigation

A deep reinforcement learning system for ultrasound image navigation with MDP-based certification analysis.

## Features

- **Segmentation**: ResNet-based U-Net for ROI detection
- **Navigation**: DQN agent for target-reaching navigation
- **MDP Analysis**: Feasibility and robustness certification
- **Web Interface**: Interactive visualization and simulation

## Quick Start

```bash
# Install dependencies
pip install uv
uv pip install -e .

# Run the API
python api.py

# Run the web app
cd webapp && npm install && npm run dev
```

## Deployment

See [DEPLOY.md](DEPLOY.md) for deployment instructions.

## License

MIT
