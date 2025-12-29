import random

# 1. Policy Evaluation (faster in-place version)
def policy_evaluation(env, policy, gamma=0.99, theta=0.01, max_iters=200):
    V = {s: 0.0 for s in env.all_states()}
    logs = []
    for it in range(max_iters):
        delta = 0.0
        for s in env.all_states():
            old_v = V[s]
            v = 0.0
            for a in policy[s]:
                for next_s, r, prob in env.transitions(s, a):
                    v += policy[s][a] * prob * (r + gamma * V.get(next_s, 0.0))
            V[s] = v
            delta = max(delta, abs(old_v - v))
        logs.append(f"Iter {it+1}: Delta = {delta:.4f}")
        if delta < theta:
            break
    return V, logs

# 2. Policy Improvement
def policy_improvement(env, V, gamma=0.99):
    policy = {s: {} for s in env.all_states()}
    logs = []
    for s in env.all_states():
        q = {}
        for a in range(len(env.actions)):
            q[a] = 0.0
            for next_s, r, prob in env.transitions(s, a):
                q[a] += prob * (r + gamma * V.get(next_s, 0.0))
        best_a = max(q, key=q.get)
        policy[s] = {a: 1.0 if a == best_a else 0.0 for a in range(len(env.actions))}
        logs.append(f"Improved policy at {s} to action {best_a}")
    return policy, logs

# 3. Policy Iteration
def policy_iteration(env, gamma=0.99, theta=0.01):
    policy = {s: {a: 1.0 / len(env.actions) for a in range(len(env.actions))} for s in env.all_states()}
    logs = []
    for _ in range(50):
        V, eval_logs = policy_evaluation(env, policy, gamma, theta)
        logs.extend(eval_logs)
        new_policy, imp_logs = policy_improvement(env, V, gamma)
        logs.extend(imp_logs)
        if new_policy == policy:
            break
        policy = new_policy
    return V, policy, logs

# 4. Value Iteration (fast in-place)
def value_iteration(env, gamma=0.99, theta=0.01, max_iters=200):
    V = {s: 0.0 for s in env.all_states()}
    logs = []
    for it in range(max_iters):
        delta = 0.0
        for s in env.all_states():
            old_v = V[s]
            max_q = -float('inf')
            for a in range(len(env.actions)):
                q = 0.0
                for next_s, r, prob in env.transitions(s, a):
                    q += prob * (r + gamma * V.get(next_s, 0.0))
                if q > max_q:
                    max_q = q
            V[s] = max_q
            delta = max(delta, abs(old_v - V[s]))
        logs.append(f"Iter {it+1}: Delta = {delta:.4f}")
        if delta < theta:
            break
    policy, imp_logs = policy_improvement(env, V, gamma)
    logs.extend(imp_logs)
    return V, policy, logs

# 5. Monte Carlo Prediction - First Visit
def mc_prediction(env, policy, gamma=0.99, episodes=100):
    V = {s: 0.0 for s in env.all_states()}
    returns = {s: [] for s in env.all_states()}
    logs = []
    for ep in range(episodes):
        episode = []
        s = env.reset()
        while True:
            a = random.choices(list(policy[s].keys()), weights=policy[s].values())[0]
            next_s, r, done, _ = env.step(a)
            episode.append((s, r))
            s = next_s
            if done:
                break
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = r + gamma * G
            if s not in visited:
                returns[s].append(G)
                V[s] = sum(returns[s]) / len(returns[s])
                visited.add(s)
                logs.append(f"Ep {ep+1}: V({s}) = {V[s]:.3f}")
    return V, logs

# 6. e-greedy MC
def epsilon_greedy_mc(env, gamma=0.99, epsilon=0.1, episodes=100):
    Q = {(s, a): 0.0 for s in env.all_states() for a in range(len(env.actions))}
    returns = {(s, a): [] for s in env.all_states() for a in range(len(env.actions))}
    logs = []
    for ep in range(episodes):
        episode = []
        s = env.reset()
        while True:
            a = random.randint(0, len(env.actions)-1) if random.random() < epsilon else max(range(len(env.actions)), key=lambda aa: Q[(s, aa)])
            next_s, r, done, _ = env.step(a)
            episode.append((s, a, r))
            s = next_s
            if done:
                break
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) not in visited:
                returns[(s, a)].append(G)
                Q[(s, a)] = sum(returns[(s, a)]) / len(returns[(s, a)])
                visited.add((s, a))
                logs.append(f"Ep {ep+1}: Q({s},{a}) = {Q[(s,a)]:.3f}")
    policy = {s: max(range(len(env.actions)), key=lambda a: Q[(s, a)]) for s in env.all_states()}
    return Q, policy, logs

# 7. n-step TD
def n_step_td(env, policy, gamma=0.99, alpha=0.1, n=3, episodes=100):
    V = {s: 0.0 for s in env.all_states()}
    logs = []
    for ep in range(episodes):
        s = env.reset()
        T = float('inf')
        rewards = []
        states = [s]
        t = 0
        while True:
            if t < T:
                a = random.choices(list(policy[s].keys()), weights=policy[s].values())[0]
                next_s, r, done, _ = env.step(a)
                rewards.append(r)
                states.append(next_s)
                if done:
                    T = t + 1
                s = next_s
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**i * rewards[tau + i] for i in range(min(n, T - tau)))
                if tau + n < T:
                    G += gamma**n * V.get(states[tau + n], 0.0)
                V[states[tau]] += alpha * (G - V[states[tau]])
                logs.append(f"Ep {ep+1}: V({states[tau]}) = {V[states[tau]]:.3f}")
            t += 1
            if tau == T - 1:
                break
    return V, logs

# 8. TD(0)
def td_zero(env, policy, gamma=0.99, alpha=0.1, episodes=100):
    V = {s: 0.0 for s in env.all_states()}
    logs = []
    for ep in range(episodes):
        s = env.reset()
        while True:
            a = random.choices(list(policy[s].keys()), weights=policy[s].values())[0]
            next_s, r, done, _ = env.step(a)
            target = r + gamma * V.get(next_s, 0.0) if not done else r
            V[s] += alpha * (target - V[s])
            logs.append(f"Ep {ep+1}: V({s}) = {V[s]:.3f}")
            s = next_s
            if done:
                break
    return V, logs

# 9. SARSA
def sarsa(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=100):
    Q = {(s, a): 0.0 for s in env.all_states() for a in range(len(env.actions))}
    logs = []
    for ep in range(episodes):
        s = env.reset()
        a = random.randint(0, len(env.actions)-1) if random.random() < epsilon else max(range(len(env.actions)), key=lambda aa: Q[(s, aa)])
        while True:
            next_s, r, done, _ = env.step(a)
            if done:
                Q[(s, a)] += alpha * (r - Q[(s, a)])
                break
            next_a = random.randint(0, len(env.actions)-1) if random.random() < epsilon else max(range(len(env.actions)), key=lambda aa: Q[(next_s, aa)])
            target = r + gamma * Q.get((next_s, next_a), 0.0)
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            logs.append(f"Ep {ep+1}: Q({s},{a}) = {Q[(s,a)]:.3f}")
            s, a = next_s, next_a
    policy = {s: max(range(len(env.actions)), key=lambda a: Q[(s, a)]) for s in env.all_states()}
    return Q, policy, logs

# 10. Q-Learning
def q_learning(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=100):
    Q = {(s, a): 0.0 for s in env.all_states() for a in range(len(env.actions))}
    logs = []
    for ep in range(episodes):
        s = env.reset()
        while True:
            a = random.randint(0, len(env.actions)-1) if random.random() < epsilon else max(range(len(env.actions)), key=lambda aa: Q[(s, aa)])
            next_s, r, done, _ = env.step(a)
            if done:
                Q[(s, a)] += alpha * (r - Q[(s, a)])
                break
            max_next = max(Q.get((next_s, aa), 0.0) for aa in range(len(env.actions)))
            target = r + gamma * max_next
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            logs.append(f"Ep {ep+1}: Q({s},{a}) = {Q[(s,a)]:.3f}")
            s = next_s
    policy = {s: max(range(len(env.actions)), key=lambda a: Q[(s, a)]) for s in env.all_states()}
    return Q, policy, logs