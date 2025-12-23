"""
FastAPI Backend for Ultrasound RL Navigation MDP Analysis

Provides:
- /simulate: Run rollouts and return trajectories + transitions
- /analyze: Perform MDP feasibility and robustness analysis
"""

import os
import tempfile

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.rollout_runner import run_rollouts
from core.mdp_analysis import full_mdp_analysis


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


app = FastAPI(
    title="Robotic Surgery Plan Certification System",
    description="MDP-based analysis for certifiable robotic surgery policies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "ultrasound-rl-navigation"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Ultrasound RL Navigation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/simulate": "POST - Run rollouts and MDP analysis",
        }
    }


@app.post("/simulate")
async def simulate(
    image: UploadFile = File(...),
    rollout_num: int = Form(100),
    stochastic_policy: str = Form("true"),
    temperature: float = Form(1.0),
    max_steps: int = Form(50),
    success_only: str = Form("false"),
    state_mode: str = Form("grid"),  # "grid" or "distance"
    target_radius: int = Form(10),   # radius for distance-based target state
    slip_prob: float = Form(0.0),    # environment stochasticity (0 = deterministic)
):
    """
    Run rollouts and return trajectories + transition matrix.
    Also performs MDP analysis on the collected transitions.
    
    Args:
        success_only: If "true", only use successful rollouts for analysis
        state_mode: "grid" for grid-based states, "distance" for view-based with distance threshold
        target_radius: for distance mode, positions within this radius are target state
    """
    # Parse boolean strings
    stochastic_policy_bool = stochastic_policy.lower() == "true"
    success_only_bool = success_only.lower() == "true"
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run rollouts
        rollout_result = run_rollouts(
            image_path=tmp_path,
            segmentation_model_path="./results/trained_models/simple_resnet_unet_best.pth",
            navigation_model_path="./results/trained_models/agent_final.pt",
            num_rollouts=rollout_num,
            stochastic_policy=stochastic_policy_bool,
            temperature=temperature,
            max_steps=max_steps,
            state_mode=state_mode,
            target_radius=target_radius,
            slip_prob=slip_prob,
        )
        
        # Get target state and num_states
        target_state = rollout_result['target_state']
        num_states = rollout_result['num_states']
        all_trajectories = rollout_result['all_trajectories']
        
        # Filter trajectories if success_only is True
        if success_only_bool:
            filtered_trajectories = [t for t in all_trajectories if t['reached_target']]
            
            # Rebuild transition matrix from filtered trajectories only
            filtered_transitions = np.zeros((num_states, num_states), dtype=np.int32)
            
            for traj in filtered_trajectories:
                steps = traj['steps']
                for i in range(len(steps) - 1):
                    from_state = steps[i]['state']
                    to_state = steps[i + 1]['state']
                    if 0 <= from_state < num_states and 0 <= to_state < num_states:
                        filtered_transitions[from_state][to_state] += 1
        else:
            # Use original transition matrix directly
            filtered_trajectories = all_trajectories
            filtered_transitions = np.array(rollout_result['transitions'])
        
        # Debug: check target row in transition matrix
        target_row = filtered_transitions[target_state]
        target_row_sum = np.sum(target_row)
        print(f"[api] Target state: {target_state}")
        print(f"[api] Target row sum: {target_row_sum}")
        if target_row_sum > 0:
            print(f"[api] WARNING: Target has outgoing transitions!")
            nonzero = np.nonzero(target_row)[0]
            print(f"[api] Target successors: {nonzero.tolist()}")
        
        # Perform MDP analysis (pass ny for physical distance calculations)
        ny = rollout_result.get('ny', 10)
        analysis_result = full_mdp_analysis(filtered_transitions, target_state, ny=ny)
        
        # Compute filtered statistics
        filtered_successful = sum(1 for t in filtered_trajectories if t['reached_target'])
        
        # Merge results
        result = {
            **rollout_result,
            'transitions': filtered_transitions.tolist(),
            'analysis': analysis_result,
            'filter_applied': success_only_bool,
            'filtered_rollout_count': len(filtered_trajectories),
            'filtered_successful_rollouts': filtered_successful,
            # Debug info
            'debug': {
                'target_state': target_state,
                'target_row_sum': int(target_row_sum),
                'target_successors': np.nonzero(target_row)[0].tolist() if target_row_sum > 0 else [],
            }
        }
        
        # Convert all numpy types to Python native types for JSON serialization
        result = convert_numpy_types(result)
        
    finally:
        os.remove(tmp_path)

    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Robotic Surgery Plan Certification System"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8765, reload=True)
