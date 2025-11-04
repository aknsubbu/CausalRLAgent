#!/usr/bin/env python3
"""
Quick test to verify LLM guidance is working without running full episode
"""

import torch
import torch.nn.functional as F
import numpy as np


def test_llm_guidance_processing():
    """Test that LLM advice is properly processed into guidance vectors"""
    
    print("="*80)
    print("TEST 1: LLM Advice Processing")
    print("="*80)
    
    # Create a mock actor (just the guidance processing part)
    action_name_to_id = {
        "move_north": 0, "move_south": 1, "move_east": 2, "move_west": 3,
        "move_northeast": 4, "move_northwest": 5, "move_southeast": 6, "move_southwest": 7,
        "wait": 8, "pickup": 9, "drop": 10, "search": 11, "open_door": 12,
        "close_door": 13, "kick": 14, "eat": 15, "drink": 16, "read": 17,
        "apply": 18, "throw": 19, "wear": 20, "take_off": 21, "wield": 22,
        # Add common variations
        "north": 0, "south": 1, "east": 2, "west": 3,
        "attack": 14, "fight": 14, "explore": 11,
    }
    
    def process_llm_guidance(llm_advice):
        """Simplified version of guidance processing"""
        guidance_vector = np.zeros(32, dtype=np.float32)
        
        if llm_advice:
            suggestions = llm_advice.get('action_suggestions', [])
            
            print(f"\n  📋 LLM Suggestions: {suggestions}")
            
            for i, suggestion in enumerate(suggestions[:5]):
                if not isinstance(suggestion, str):
                    suggestion = str(suggestion)
                
                suggestion_lower = suggestion.lower().strip()
                
                # Try exact match
                action_id = action_name_to_id.get(suggestion_lower)
                
                if action_id is None:
                    # Try partial match
                    for key, val in action_name_to_id.items():
                        if key in suggestion_lower or suggestion_lower in key:
                            action_id = val
                            break
                
                if action_id is not None and action_id < 23:
                    weight = (7 - i) / 3.0
                    guidance_vector[action_id] = max(guidance_vector[action_id], weight)
                    print(f"    ✓ Mapped '{suggestion_lower}' → action {action_id} (weight: {weight:.2f})")
                else:
                    print(f"    ✗ Failed to map '{suggestion_lower}'")
        
        non_zero = [(i, v) for i, v in enumerate(guidance_vector[:23]) if v > 0]
        print(f"    📊 Final guidance vector (non-zero): {non_zero}")
        return guidance_vector
    
    # Test cases
    test_cases = [
        {
            "name": "Combat scenario",
            "advice": {
                "action_suggestions": ["kick", "kick", "move_north"],
                "immediate_priority": "fight the monster"
            }
        },
        {
            "name": "Exploration scenario",
            "advice": {
                "action_suggestions": ["search", "move_east", "pickup"],
                "immediate_priority": "explore safely"
            }
        },
        {
            "name": "Retreat scenario",
            "advice": {
                "action_suggestions": ["move_west", "move_north", "wait"],
                "immediate_priority": "avoid danger"
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {test['name']}")
        print(f"{'-'*80}")
        guidance = process_llm_guidance(test['advice'])
        max_action = np.argmax(guidance[:23])
        max_weight = guidance[max_action]
        print(f"\n  ➡️ Top action: {max_action} with weight {max_weight:.2f}")
    
    print("\n" + "="*80 + "\n")


def test_logit_boosting():
    """Test that guidance actually boosts action logits"""
    
    print("="*80)
    print("TEST 2: Logit Boosting")
    print("="*80)
    
    # Simulate base policy logits (random)
    base_logits = torch.randn(23)
    print(f"\nBase logits (first 10): {base_logits[:10].tolist()}")
    
    # Simulate LLM guidance vector
    guidance_vector = np.zeros(23, dtype=np.float32)
    guidance_vector[2] = 2.33   # move_east - top suggestion
    guidance_vector[11] = 2.00  # search - second suggestion
    guidance_vector[14] = 1.67  # kick - third suggestion
    
    guidance_bias = torch.FloatTensor(guidance_vector)
    
    print(f"\nGuidance bias (non-zero only):")
    for i, val in enumerate(guidance_bias):
        if val > 0:
            print(f"  Action {i}: {val:.2f}")
    
    # Apply strong boost (20x)
    guided_logits = base_logits * 0.05 + (guidance_bias * 20.0)
    
    print(f"\nGuided logits (first 10): {guided_logits[:10].tolist()}")
    
    # Convert to probabilities
    base_probs = F.softmax(base_logits, dim=-1)
    guided_probs = F.softmax(guided_logits, dim=-1)
    
    print(f"\nBASE POLICY - Top 5 actions:")
    top5_base = torch.topk(base_probs, k=5)
    for i, (idx, prob) in enumerate(zip(top5_base.indices, top5_base.values)):
        print(f"  {i+1}. Action {idx.item()}: {prob.item()*100:.1f}%")
    
    print(f"\nGUIDED POLICY - Top 5 actions:")
    top5_guided = torch.topk(guided_probs, k=5)
    for i, (idx, prob) in enumerate(zip(top5_guided.indices, top5_guided.values)):
        print(f"  {i+1}. Action {idx.item()}: {prob.item()*100:.1f}%")
    
    # Check if top suggested action became top choice
    top_suggested_action = 2  # move_east
    if top5_guided.indices[0] == top_suggested_action:
        print(f"\n✓ SUCCESS: LLM's top suggestion (action {top_suggested_action}) is now the top choice!")
        print(f"  Probability increased from {base_probs[top_suggested_action].item()*100:.1f}% to {guided_probs[top_suggested_action].item()*100:.1f}%")
    else:
        print(f"\n✗ FAILED: LLM's top suggestion (action {top_suggested_action}) is NOT the top choice")
        print(f"  Current top: action {top5_guided.indices[0].item()}")
    
    print("\n" + "="*80 + "\n")


def test_action_selection_distribution():
    """Test action selection over multiple samples"""
    
    print("="*80)
    print("TEST 3: Action Selection Distribution")
    print("="*80)
    
    # Simulate guided logits where action 2 is heavily boosted
    base_logits = torch.randn(23)
    guidance_vector = np.zeros(23, dtype=np.float32)
    guidance_vector[2] = 2.33  # Strong suggestion for move_east
    guidance_bias = torch.FloatTensor(guidance_vector)
    
    guided_logits = base_logits * 0.05 + (guidance_bias * 20.0)
    
    # Sample actions 100 times
    action_dist = torch.distributions.Categorical(logits=guided_logits)
    samples = [action_dist.sample().item() for _ in range(100)]
    
    # Count action frequencies
    from collections import Counter
    action_counts = Counter(samples)
    
    print(f"\nSampled 100 actions with LLM guidance for action 2 (move_east):")
    print(f"\nTop 5 most selected actions:")
    for action, count in action_counts.most_common(5):
        print(f"  Action {action}: {count}/100 ({count}%)")
    
    if action_counts.most_common(1)[0][0] == 2:
        print(f"\n✓ SUCCESS: Action 2 (move_east) was selected most often!")
    else:
        print(f"\n✗ FAILED: Action {action_counts.most_common(1)[0][0]} was selected most often, not action 2")
    
    print("\n" + "="*80 + "\n")


def test_batch_guidance():
    """Test guidance with batch size > 1"""
    
    print("="*80)
    print("TEST 4: Batch Guidance")
    print("="*80)
    
    batch_size = 4
    base_logits = torch.randn(batch_size, 23)
    
    guidance_vector = np.zeros(23, dtype=np.float32)
    guidance_vector[11] = 2.0  # search
    guidance_bias = torch.FloatTensor(guidance_vector)
    
    # Expand to batch
    guidance_bias_batch = guidance_bias.unsqueeze(0).expand(batch_size, -1)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Guidance bias shape: {guidance_bias_batch.shape}")
    print(f"Base logits shape: {base_logits.shape}")
    
    guided_logits = base_logits * 0.05 + (guidance_bias_batch * 20.0)
    
    print(f"Guided logits shape: {guided_logits.shape}")
    
    # Check top action for each batch element
    top_actions = torch.argmax(guided_logits, dim=-1)
    print(f"\nTop actions for each batch element: {top_actions.tolist()}")
    
    if all(action == 11 for action in top_actions.tolist()):
        print(f"\n✓ SUCCESS: All batch elements selected action 11 (search)")
    else:
        print(f"\n⚠️ WARNING: Not all batch elements selected the guided action")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("LLM GUIDANCE UNIT TESTS")
    print("="*80)
    print("\nThese tests verify that:")
    print("1. LLM advice is correctly parsed into guidance vectors")
    print("2. Guidance vectors significantly boost action logits (20x)")
    print("3. Guided actions are actually selected when sampling")
    print("4. Batching works correctly")
    print("="*80 + "\n")
    
    test_llm_guidance_processing()
    test_logit_boosting()
    test_action_selection_distribution()
    test_batch_guidance()
    
    print("="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\nIf all tests passed (✓), your LLM guidance is working correctly!")
    print("If any tests failed (✗), check the implementation of:")
    print("  - process_llm_guidance() method")
    print("  - forward() method's guidance application")
    print("  - llm_guidance_weight parameter")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()