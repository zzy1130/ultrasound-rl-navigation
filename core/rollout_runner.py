"""
Rollout runner: run multiple navigation episodes (rollouts) and record grid-based state transitions.
Each rollout is equivalent to running main.py --mode demo (without generating images/gifs).
"""
import os
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
from PIL import Image

from core.navigation_agent import NavigationAgent
from core.navigation_environment import NavigationEnvironment
from core.segmentation_model import load_segmentation_model
from core.utils import find_center


def _compute_center_env(image_path: str, seg_model_path: str, device) -> Tuple[Tuple[int, int], np.ndarray]:
    """Segment image and compute target center (same as main.py demo)."""
    image = Image.open(image_path).convert("L")
    orig_w, orig_h = image.size
    image_resized = image.resize((256, 256))

    seg_model = load_segmentation_model(seg_model_path, device)
    image_tensor = torch.FloatTensor(np.array(image_resized) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_tensor = seg_model(image_tensor)
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    center_resized = find_center(mask)
    if (orig_w, orig_h) != (256, 256):
        center_env = (int(center_resized[0] * orig_w / 256), int(center_resized[1] * orig_h / 256))
    else:
        center_env = center_resized
    return center_env, mask


def run_rollouts(
    image_path: str,
    segmentation_model_path: str,
    navigation_model_path: str,
    num_rollouts: int = 100,
    stochastic_policy: bool = True,
    temperature: float = 1.0,
    max_steps: int = 50,
    state_mode: str = "grid",  # "grid" or "distance"
    target_radius: int = 10,   # radius for distance-based target state
    slip_prob: float = 0.0,    # environment stochasticity (probability of random action)
) -> Dict:
    """
    Run multiple rollouts on a single image, record ALL trajectories for visualization.
    
    Each rollout is equivalent to running:
        python main.py --mode demo --image <image_path> ...
    without generating images/gifs.
    
    Args:
        state_mode: "grid" for grid-based states, "distance" for view-based with distance threshold
        target_radius: for distance mode, positions within this radius are target state
    
    Returns:
        dict with:
        - transitions: state transition count matrix
        - all_trajectories: list of trajectories, each containing positions for every step
        - statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rollout_runner] Using device: {device}")
    print(f"[rollout_runner] Running {num_rollouts} rollouts on {image_path}")
    print(f"[rollout_runner] State mode: {state_mode}, target_radius: {target_radius}, slip_prob: {slip_prob}")

    # Segment image and get target center
    center_env, mask = _compute_center_env(image_path, segmentation_model_path, device)
    print(f"[rollout_runner] Target center: {center_env}")

    # Load navigation agent
    nav_agent = NavigationAgent(num_actions=5, device=device)
    nav_agent.load(navigation_model_path)

    # Get original image size
    orig_image = Image.open(image_path)
    orig_w, orig_h = orig_image.size

    # Create environment with state mode
    centers_dict = {os.path.basename(image_path): center_env}
    env = NavigationEnvironment(
        [image_path], 
        centers_dict, 
        max_steps=max_steps,
        state_mode=state_mode,
        target_radius=target_radius,
        slip_prob=slip_prob,
    )
    
    # Initialize to get state space size
    env.reset()
    
    if state_mode == "distance":
        nx, ny, num_states = env.distance_state_size()
        print(f"[rollout_runner] Distance mode: {nx} x {ny} + 1 target = {num_states} states")
    else:
        nx, ny = env.grid_size()
        num_states = nx * ny
        print(f"[rollout_runner] Grid mode: {nx} x {ny} = {num_states} states")
    
    # Transition count matrix
    transitions = np.zeros((num_states, num_states), dtype=np.int64)
    
    # Store ALL trajectories for visualization
    all_trajectories = []
    
    # Statistics
    total_steps = 0
    successful_rollouts = 0

    # Run all rollouts
    for episode in range(num_rollouts):
        obs = env.reset()
        
        # Get initial state using unified interface
        initial_state_id = env.get_state_id(env.current_position)
        
        # Record trajectory for this rollout - include starting position
        trajectory = [{
            "step": 0,
            "position": list(env.current_position),
            "state": initial_state_id,
        }]
        
        reached_target = False
        target_state = env.get_target_state_id()
        
        for step in range(max_steps):
            # Get current state using unified interface
            state_id = env.get_state_id(env.current_position)
            
            # If already at target, break immediately (target is absorbing)
            if state_id == target_state:
                reached_target = True
                # Record final state in trajectory if not already there
                if not trajectory or trajectory[-1]['state'] != target_state:
                    trajectory.append({
                        "step": step,
                        "position": list(env.current_position),
                        "state": target_state,
                        "done": True,
                    })
                break
            
            # Agent selects action (same as main.py demo)
            action = nav_agent.act(
                obs,
                training=False,
                stochastic_policy=stochastic_policy,
                temperature=temperature,
            )
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Get next state using info (already computed in step())
            next_id = info["state_id"]
            
            # Update transition matrix (never record transitions FROM target state)
            if state_id < num_states and next_id < num_states and state_id != target_state:
                transitions[state_id, next_id] += 1
            total_steps += 1
            
            # Record step in trajectory
            trajectory.append({
                "step": step + 1,
                "position": list(info["position"]),
                "state": next_id,
                "action": action,
                "done": done,
            })
            
            obs = next_obs
            
            if done:
                reached_target = next_id == target_state
                break
        
        if reached_target:
            successful_rollouts += 1
        
        # Store this rollout's trajectory
        all_trajectories.append({
            "rollout_id": episode + 1,
            "steps": trajectory,
            "reached_target": reached_target,
            "length": len(trajectory),
        })
        
        # Progress log
        if (episode + 1) % max(1, num_rollouts // 10) == 0:
            print(f"[rollout_runner] Completed {episode + 1}/{num_rollouts} rollouts")

    # Get target state using unified interface
    target_state = env.get_target_state_id()

    print(f"[rollout_runner] Finished all {num_rollouts} rollouts")
    print(f"[rollout_runner] Total steps: {total_steps}, Avg: {total_steps/num_rollouts:.1f}")
    print(f"[rollout_runner] Success rate: {successful_rollouts}/{num_rollouts} ({100*successful_rollouts/num_rollouts:.1f}%)")
    print(f"[rollout_runner] Target state: {target_state}")

    return {
        "nx": nx,
        "ny": ny,
        "num_states": num_states,
        "transitions": transitions.tolist(),
        "target_state": target_state,
        "all_trajectories": all_trajectories,
        "total_steps": total_steps,
        "rollouts": num_rollouts,
        "successful_rollouts": successful_rollouts,
        "avg_rollout_length": total_steps / num_rollouts if num_rollouts > 0 else 0,
        "center": list(center_env),
        "image_size": [orig_w, orig_h],
        "view_size": env.view_size,
        "state_mode": state_mode,
        "target_radius": target_radius,
    }
