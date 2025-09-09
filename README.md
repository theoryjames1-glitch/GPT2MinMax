
# GPT2MinMax

A conceptual framework that applies the **minimax game theory principle** to two GPT agents, where one maximizes potential outcomes and the other minimizes them. The result is a balanced, adversarially-tested reasoning process.

---

## Overview

**GPT2MinMax** sets up two agents:

* **GPT-Max** → Proposes actions, ideas, or strategies to *maximize* utility, clarity, or effectiveness.
* **GPT-Min** → Responds with counteractions or critiques to *minimize* risks, flaws, or weaknesses.

This interaction simulates a **minimax search**, where every proposal must withstand adversarial testing before reaching equilibrium.

---

## How It Works

1. **Define State (S)**: The problem, question, or context.
2. **Maximizer Step**: GPT-Max generates the best possible action to maximize utility.
3. **Minimizer Step**: GPT-Min generates the strongest counter-response to reduce the utility of GPT-Max’s choice.
4. **Equilibrium**: The final outcome balances both forces, yielding robust, resilient reasoning.

**Formula:**

```
V(S) = max(Max’s Actions) min(Min’s Responses) U(S, Action)
```

Where **U** is the utility function measuring quality, strength, or payoff.

---

## Example

**Problem:** Best way to learn a new programming language?

* **GPT-Max:** “Build a full project from scratch to maximize immersion.”
* **GPT-Min:** “That risks overwhelming beginners—start with small exercises first.”

**Equilibrium Result:**
“Begin with small guided exercises, then transition into a larger project for deeper learning.”

---

## Why GPT2MinMax?

* **Self-Critique** → Answers survive adversarial testing.
* **Balanced Reasoning** → Avoids one-sided or overly optimistic solutions.
* **Robust Outputs** → Inspired by decision-making in chess, Go, and AI planning.

---

## Extensions

* **Multi-Agent Expansion:** Add more roles (neutral, stochastic, evaluator).
* **Expectiminimax:** Introduce randomness to model uncertainty.
* **Applications:** Ethics, negotiation, policy, strategy, education.

---

## License

This is a theoretical framework—adapt freely for research, projects, or experimentation.

### PSEUDOCODE

```python
import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Setup ===
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# === Minimax Utility ===
def V(max_text, min_text):
    """Simple minimax reward: Max stronger if it's longer than Min."""
    return len(max_text) - len(min_text)

def generate_text(prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=50256,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# === One-Prompt Forever Loop ===
def minmax_forever(initial_prompt, rounds=20):
    # Step 0: only once → seed goes into Min
    context = initial_prompt

    for t in range(1, rounds + 1):
        # Min step
        min_out = generate_text(f"Min critique:\n{context}\n")
        # Max step (input is Min's output)
        max_out = generate_text(f"Max counter:\n{min_out}\n")

        # Minimax score
        score = V(max_out, min_out)

        # Reinforcement update
        seq = min_out + max_out
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        target_ids = inputs["input_ids"].clone()

        ce_loss = loss_fn(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            target_ids[:, 1:].reshape(-1)
        )
        loss = ce_loss * (-score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Feed Max output into Min for the next round
        context = max_out

        # Trace
        print(f"\n--- Round {t} ---")
        print("Min:", min_out)
        print("Max:", max_out)
        print("V(S):", score)
        print("Loss:", loss.item())

# === Run ===
if __name__ == "__main__":
    seed = "What is the best way to learn a new programming language?"
    minmax_forever(seed, rounds=10)
```
