from flask import Flask, render_template, request
from environments import get_env, visualize_values, visualize_policy
from rl_algorithms import policy_iteration, value_iteration, mc_prediction, epsilon_greedy_mc, n_step_td, td_zero, sarsa, q_learning

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    envs = ["StochasticGridWorld", "Taxi-v3", "CartPole-v1"]
    algos = ["Policy Iteration", "Value Iteration", "MC Prediction", "e-greedy MC", "n-step TD", "TD(0)", "SARSA", "Q-Learning"]
    results = None

    if request.method == 'POST':
        env_name = request.form['env']
        algo_name = request.form['algo']
        gamma = float(request.form.get('gamma', 0.99))
        alpha = float(request.form.get('alpha', 0.1))
        epsilon = float(request.form.get('epsilon', 0.1))
        n = int(request.form.get('n', 3))
        episodes = int(request.form.get('episodes', 100))
        theta = float(request.form.get('theta', 0.01))

        env = get_env(env_name)
        # Random policy for prediction algos
        policy = {s: {a: 1.0 / len(env.actions) for a in range(len(env.actions))} for s in env.all_states()}

        if algo_name == "Policy Iteration":
            V, pol, logs = policy_iteration(env, gamma, theta)
            val_img = visualize_values(V, env, env_name)
            pol_img = visualize_policy(pol, env, env_name)
        elif algo_name == "Value Iteration":
            V, pol, logs = value_iteration(env, gamma, theta)
            val_img = visualize_values(V, env, env_name)
            pol_img = visualize_policy(pol, env, env_name)
        elif algo_name == "MC Prediction":
            V, logs = mc_prediction(env, policy, gamma, episodes)
            val_img = visualize_values(V, env, env_name)
            pol_img = ""
        elif algo_name == "e-greedy MC":
            Q, pol, logs = epsilon_greedy_mc(env, gamma, epsilon, episodes)
            val_img = ""
            pol_img = visualize_policy(pol, env, env_name)
        elif algo_name == "n-step TD":
            V, logs = n_step_td(env, policy, gamma, alpha, n, episodes)
            val_img = visualize_values(V, env, env_name)
            pol_img = ""
        elif algo_name == "TD(0)":
            V, logs = td_zero(env, policy, gamma, alpha, episodes)
            val_img = visualize_values(V, env, env_name)
            pol_img = ""
        elif algo_name == "SARSA":
            Q, pol, logs = sarsa(env, gamma, alpha, epsilon, episodes)
            val_img = ""
            pol_img = visualize_policy(pol, env, env_name)
        elif algo_name == "Q-Learning":
            Q, pol, logs = q_learning(env, gamma, alpha, epsilon, episodes)
            val_img = ""
            pol_img = visualize_policy(pol, env, env_name)

        results = {
            'val_img': val_img,
            'pol_img': pol_img,
            'logs': '\n'.join(logs[:50]) if 'logs' in locals() else "Completed."
        }

    return render_template('index.html', envs=envs, algos=algos, results=results)

if __name__ == '__main__':
    app.run(debug=True)