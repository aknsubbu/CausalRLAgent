#!/usr/bin/env python3
"""
Verification script to test if LLM advice is actually being followed
Run this AFTER applying the fixes
"""

import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict

class LLMGuidanceVerifier:
    """Verify that LLM guidance is actually affecting agent behavior"""
    
    def __init__(self):
        self.llm_suggestions = []
        self.actual_actions = []
        self.action_probs_with_llm = []
        self.action_probs_without_llm = []
        
    def log_llm_step(self, llm_advice, action_taken, action_probs):
        """Log a step where LLM provided advice"""
        if llm_advice and 'action_suggestions' in llm_advice:
            suggestions = llm_advice['action_suggestions']
            self.llm_suggestions.append(suggestions)
            self.actual_actions.append(action_taken)
            self.action_probs_with_llm.append(action_probs)
    
    def analyze_compliance(self):
        """Analyze how often agent follows LLM advice"""
        if not self.llm_suggestions:
            print("No LLM steps logged yet!")
            return
        
        print("\n" + "="*80)
        print("LLM GUIDANCE COMPLIANCE ANALYSIS")
        print("="*80)
        
        # Create action name mapping
        action_names = {
            0: "move_north", 1: "move_south", 2: "move_east", 3: "move_west",
            4: "move_northeast", 5: "move_northwest", 6: "move_southeast", 7: "move_southwest",
            8: "wait", 9: "pickup", 10: "drop", 11: "search", 12: "open_door",
            13: "close_door", 14: "kick", 15: "eat", 16: "drink", 17: "read",
            18: "apply", 19: "throw", 20: "wear", 21: "take_off", 22: "wield"
        }
        
        # Map action names to IDs
        name_to_id = {v: k for k, v in action_names.items()}
        
        top_1_matches = 0
        top_3_matches = 0
        
        for i, (suggestions, action) in enumerate(zip(self.llm_suggestions, self.actual_actions)):
            action_name = action_names.get(action, f"action_{action}")
            
            # Try to map LLM suggestions to action IDs
            suggestion_ids = []
            for sug in suggestions[:3]:
                if isinstance(sug, str):
                    sug_lower = sug.lower().strip()
                    # Try direct match
                    if sug_lower in name_to_id:
                        suggestion_ids.append(name_to_id[sug_lower])
                    # Try partial match
                    else:
                        for name, id in name_to_id.items():
                            if name in sug_lower or sug_lower in name:
                                suggestion_ids.append(id)
                                break
            
            # Check if action matches top suggestion
            if suggestion_ids and action == suggestion_ids[0]:
                top_1_matches += 1
                print(f"  ✓ Step {i}: Followed top suggestion '{suggestions[0]}' → {action_name}")
            elif action in suggestion_ids:
                top_3_matches += 1
                print(f"  ~ Step {i}: Followed suggestion #{suggestion_ids.index(action)+1} '{suggestions[suggestion_ids.index(action)]}' → {action_name}")
            else:
                print(f"  ✗ Step {i}: IGNORED advice {suggestions[:3]} → took {action_name} instead")
        
        compliance_rate_top1 = (top_1_matches / len(self.llm_suggestions)) * 100
        compliance_rate_top3 = ((top_1_matches + top_3_matches) / len(self.llm_suggestions)) * 100
        
        print(f"\n{'='*80}")
        print(f"Top-1 Compliance: {top_1_matches}/{len(self.llm_suggestions)} ({compliance_rate_top1:.1f}%)")
        print(f"Top-3 Compliance: {top_1_matches + top_3_matches}/{len(self.llm_suggestions)} ({compliance_rate_top3:.1f}%)")
        print(f"{'='*80}\n")
        
        if compliance_rate_top1 < 30:
            print("⚠️  WARNING: Compliance is very low! LLM guidance may not be working.")
            print("   Expected: >60% top-1 compliance with fixes applied")
        elif compliance_rate_top1 < 60:
            print("⚠️  Moderate compliance. LLM guidance is partially working.")
            print("   Target: >60% top-1 compliance")
        else:
            print("✓ Good compliance! LLM guidance is working as expected.")
    
    def analyze_probability_shift(self):
        """Analyze how much LLM shifts action probabilities"""
        if not self.action_probs_with_llm:
            print("No probability data logged yet!")
            return
        
        print("\n" + "="*80)
        print("ACTION PROBABILITY SHIFT ANALYSIS")
        print("="*80)
        
        for i, probs in enumerate(self.action_probs_with_llm[:5]):  # Show first 5
            top_5_actions = np.argsort(probs)[-5:][::-1]
            print(f"\nStep {i} - Top 5 action probabilities:")
            for rank, action_id in enumerate(top_5_actions):
                prob = probs[action_id]
                print(f"  {rank+1}. Action {action_id}: {prob:.4f} ({prob*100:.1f}%)")
        
        print(f"\n{'='*80}\n")


async def run_verification_episode(agent, env, verifier, max_steps=100):
    """Run a single episode with verification logging"""
    
    print("\n" + "="*80)
    print("STARTING VERIFICATION EPISODE")
    print("="*80 + "\n")
    
    obs = env.reset()
    agent.actor.reset_hidden_states()
    agent.critic.reset_hidden_states()
    
    llm_call_count = 0
    
    for step in range(max_steps):
        # Check if LLM should be called
        if agent.llm_advisor.should_call_llm():
            # Get semantic description
            tensor_obs, processed_obs = agent.process_observation(obs)
            semantic_desc = agent.semantic_descriptor.generate_full_description(
                obs, processed_obs, agent.recent_actions
            )
            
            # Get LLM advice
            recent_performance = {
                'avg_reward': 0,
                'avg_length': step,
                'death_rate': 0
            }
            
            llm_advice = await agent.llm_advisor.get_strategic_advice(
                semantic_desc, recent_performance
            )
            
            agent.current_llm_advice = llm_advice
            llm_call_count += 1
            
            print(f"\n{'─'*80}")
            print(f"Step {step}: LLM ADVICE #{llm_call_count}")
            print(f"{'─'*80}")
            print(f"Priority: {llm_advice.get('immediate_priority', 'N/A')}")
            print(f"Suggestions: {llm_advice.get('action_suggestions', [])}")
        
        # Select action
        tensor_obs, processed_obs = agent.process_observation(obs)
        
        with torch.no_grad():
            # Reset hidden states for first step
            reset_hidden = (step == 0)
            action_logits = agent.actor(tensor_obs, reset_hidden, agent.current_llm_advice)
            
            # Handle both 1D and 2D logits
            if action_logits.dim() == 1:
                action_probs = F.softmax(action_logits, dim=-1).cpu().numpy()
            else:
                action_probs = F.softmax(action_logits, dim=-1).cpu().numpy()[0]
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
        
        # Log if LLM was active
        if agent.current_llm_advice:
            verifier.log_llm_step(
                agent.current_llm_advice,
                action.item(),
                action_probs
            )
        
        # Take environment step
        step_result = env.step(action.item())
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Analyze results
    verifier.analyze_compliance()
    verifier.analyze_probability_shift()


async def main_verification():
    """Main verification function"""
    print("="*80)
    print("LLM GUIDANCE VERIFICATION TOOL")
    print("="*80)
    print("\nThis script will:")
    print("1. Run one episode with your agent")
    print("2. Log all LLM advice and actions taken")
    print("3. Calculate compliance rate (how often agent follows advice)")
    print("4. Show probability distributions")
    print("\nExpected results AFTER fixes:")
    print("  - Top-1 Compliance: >60%")
    print("  - Top-3 Compliance: >80%")
    print("  - Top action probability: >30% when LLM active")
    print("="*80 + "\n")
    
    # Import your agent
    from llm_guided_v2 import (  # Replace with actual script name
        create_nethack_env,
        MonitoredLLMGuidedNetHackAgent
    )
    
    # Create environment and agent
    env = create_nethack_env()
    agent = MonitoredLLMGuidedNetHackAgent(
        action_dim=env.action_space.n,
        llm_guidance_weight=0.95,
        llm_call_frequency=10
    )
    
    # Create verifier
    verifier = LLMGuidanceVerifier()
    
    # Run verification episode
    await run_verification_episode(agent, env, verifier, max_steps=100)
    
    env.close()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nIf compliance is still low (<40%), check:")
    print("1. Is process_llm_guidance() being called? Add print statements")
    print("2. Are action names being mapped correctly? Check debug output")
    print("3. Is the guidance_bias actually being added? Print guidance_bias values")
    print("4. Is llm_guidance_weight = 0.95? Print it in forward()")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main_verification())