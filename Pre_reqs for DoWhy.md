# Pre-requisites for DoWhy

Advice from the LLM-layer

```python
def get_llm_advice(observation):
    prompt = f"Given the current dungeon state: {observation[:100]}... what should the agent do next?"
    advice = call_llm(prompt)  # pseudo function
    return advice
```

Convert advice to an action vector (e.g., mapping words → discrete actions)

```python
def interpret_advice(advice_text):
    if "move down" in advice_text:
        return env.actions.index("MOVE_DOWN")
    elif "attack" in advice_text:
        return env.actions.index("ATTACK")
    else:
        return env.action_space.sample()  # fallback
```

Data Logging for DoWhy

```python
from collections import deque
import pandas as pd

history = deque(maxlen=5000)

for step in range(1000):
    obs = env.render()  # or structured obs
    advice_text = get_llm_advice(obs)
    advice_action = interpret_advice(advice_text)
    rl_action, _ = agent.predict(obs)

    # Decide what to execute (we’ll use DoWhy to filter later)
    executed_action = advice_action  # temporarily follow LLM

    new_obs, reward, done, trunc, info = env.step(executed_action)

    history.append({
        "advice_action": advice_action,
        "rl_action": rl_action,
        "reward": reward,
        "hp": info["blstats"][10],
        "depth": info["blstats"][12],
        "advice_text": advice_text
    })

    if done or trunc:
        obs, info = env.reset()

df = pd.DataFrame(history)
df.to_csv("dowhy_data.csv", index=False)
```

DoWhy Consistency Check

```python
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment="advice_action",  # treated = following LLM
    outcome="reward",
    graph='''
        digraph {
            advice_action -> reward;
            rl_action -> reward;
            hp -> reward;
            depth -> reward;
        }
    '''
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print("Causal Effect of LLM advice:", estimate.value)
```

Causal Filter Implementation

```python
if estimate.value < 0:
    print("⚠️ LLM advice not causally consistent — ignoring")
    final_action = rl_action
else:
    final_action = advice_action
```
