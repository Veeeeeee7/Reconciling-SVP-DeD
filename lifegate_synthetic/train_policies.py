import numpy as np
import pickle
from lifegate import LifeGate
from svp import value_iter, value_iter_near_greedy, V2Q

def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):
            s = y * width + x
            P[s] = {}
            if [x, y] in env.barriers:
                continue
            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward, done = 0.0, False

                # dead-end special drag
                if [x, y] in env.dead_ends:
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types in ('death','regular'): reward = -1.0
                    P[s][a].append((deadend_threshold, s + 1, reward, done))
                    P[s][a].append((1 - deadend_threshold, s, 0.0, False))
                    continue

                # terminal states
                if [new_x, new_y] in env.deaths:
                    if types in ('death','regular'): reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if [new_x, new_y] in env.recoveries:
                    if types in ('recovery','regular'): reward = +1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue

                # normal move
                if a==1:   new_y -=1
                elif a==2: new_y +=1
                elif a==3: new_x -=1
                elif a==4: new_x +=1

                # bounce off walls/barriers
                if (new_x<0 or new_y<0 or new_x>=width or new_y>=height or [new_x,new_y] in env.barriers):
                    new_x, new_y = x, y

                s_next = new_y*width + new_x
                s_drag = s + 1
                reward_drag, done_drag = 0.0, False

                if [new_x,new_y] in env.deaths:
                    done = True
                    if types in ('death','regular'): reward = -1.0
                elif [new_x,new_y] in env.recoveries:
                    done = True
                    if types in ('recovery','regular'): reward = +1.0

                if [x+1,y] in env.deaths:
                    done_drag = True
                    if types in ('death','regular'): reward_drag = -1.0

                P[s][a].append((1-env.death_drag, s_next, reward, done))
                P[s][a].append((  env.death_drag,   s_drag, reward_drag, done_drag))
    return P

def bad_policies(Q_d, env_death, env, threshold):
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=threshold)
    env_death.nS = env_death.scr_w * env_death.scr_h
    env_death.nA = env_death.nb_actions
    pi_bad = np.zeros((env_death.nS, env_death.nA), dtype=int)
    for s in range(env_death.nS):
        for a in range(env_death.nA):
            if Q_d[s,a] <= -threshold:
                pi_bad[s,a] = 1
    return pi_bad

def fraction_conflict(pi_svp, pi_bad, valid_states):
    conflicts = 0
    for s in range(pi_bad.shape[0]):
        if any(pi_svp[s,a] and pi_bad[s,a] for a in range(pi_bad.shape[1])):
            conflicts += 1
    return conflicts / valid_states

def cleaned(pi, lifegate_states, dead_states, dead_ends):
    for s in range(pi.shape[0]):
        if s in lifegate_states or s in dead_states or s in dead_ends:
            pi[s,:] = 0
    return pi

def train_search_pair(
    env, env_death, gamma,
    zeta_vals, dt_vals,
    barrier_states, lifegate_states, dead_states, dead_ends,
    theta=1e-10, max_iter=1000
):
    n_dt, n_z = len(dt_vals), len(zeta_vals)
    incons_map    = np.zeros((n_dt, n_z))
    svp_size_map  = np.zeros((n_dt, n_z))
    bad_size_map  = np.zeros((n_dt, n_z))
    total_states  = env.scr_w * env.scr_h
    valid_states  = total_states - len(barrier_states)

    # initialize regular MDP + optimal value function for SVP
    env.P = MDP_lifegate(env, types='regular', deadend_threshold=0.7)
    env.nS, env.nA = env.scr_w * env.scr_h, env.nb_actions
    V_star, _ = value_iter(env, gamma, theta=theta)
    lst_pi_svp = []

    # compute SVP policies for each zeta
    for zeta in zeta_vals:
        V_svp, pi_svp, _, _ = value_iter_near_greedy(
            env, gamma, zeta, V_star,
            theta=theta, max_iter=max_iter)
        pi_svp = cleaned(pi_svp, lifegate_states, dead_states, dead_ends)
        lst_pi_svp.append(pi_svp)

    # initialize death MDP + optimal value function for DeD
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=0.7)
    env_death.nS, env_death.nA = env_death.scr_w * env_death.scr_h, env_death.nb_actions
    V_d, _ = value_iter(env_death, gamma, theta=theta)
    Q_d = V2Q(env_death, V_d, gamma)
    lst_pi_bad = []

    # compute DeD policies for each deadend threshold
    for dt in dt_vals:
        pi_bad = bad_policies(Q_d, env_death, env, dt)
        lst_pi_bad.append(pi_bad)


    incons_tensor = np.zeros((env.nS, len(dt_vals), len(zeta_vals)), dtype=int)
    combined_size_count = 0

    # loop over all pairs of (dt, zeta) and compute inconsistency metrics
    for i, dt in enumerate(dt_vals):
        for j, zeta in enumerate(zeta_vals):
            # fill maps
            pi_bad = lst_pi_bad[i]
            pi_svp = lst_pi_svp[j]
            incons_map[i, j] = fraction_conflict(pi_svp, pi_bad, valid_states)
            svp_size_map[i, j] = pi_svp.sum(axis=1).mean()
            bad_size_map[i, j] = pi_bad.sum(axis=1).mean()
            for s in range(env.nS):
                if (len(pi_svp[s]) + len(pi_bad[s]) < 5):
                    combined_size_count += 1
                for a in range(env.nA):
                    if pi_svp[s, a] == 1 and pi_bad[s, a] == 1:
                        incons_tensor[s, i, j] = 1
                        break

    return {
        'zeta_vals': zeta_vals,
        'dt_vals':   dt_vals,
        'incons_map':    incons_map,
        'svp_size_map':  svp_size_map,
        'bad_size_map':  bad_size_map,
        'incons_tensor': incons_tensor,
        'combined_size_count': combined_size_count
    }


def combined(
    env, env_death, gamma,
    zeta_vals, dt_vals,
    barrier_states, lifegate_states, dead_states, dead_ends,
    theta=1e-10, max_iter=1000
):
    n_dt, n_z = len(dt_vals), len(zeta_vals)

    # Prepare the two policy lists
    # ——————————————————————————————————————————————————————————————
    # Regular MDP for SVP
    env.P = MDP_lifegate(env, types='regular', deadend_threshold=0.7)
    env.nS, env.nA = env.scr_w * env.scr_h, env.nb_actions
    V_star, _ = value_iter(env, gamma, theta=theta)
    lst_pi_svp = []

    # compute SVP policies for each zeta
    for zeta in zeta_vals:
        V_svp, pi_svp, _, _ = value_iter_near_greedy(
            env, gamma, zeta, V_star,
            theta=theta, max_iter=max_iter
        )
        pi_svp = cleaned(pi_svp, lifegate_states, dead_states, dead_ends)
        lst_pi_svp.append(pi_svp)

    # Death‐only MDP for DeD
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=0.7)
    env_death.nS, env_death.nA = env_death.scr_w * env_death.scr_h, env_death.nb_actions
    V_d, _ = value_iter(env_death, gamma, theta=theta)
    Q_d = V2Q(env_death, V_d, gamma)
    lst_pi_bad = []

    # compute DeD policies for each deadend threshold
    for dt in dt_vals:
        lst_pi_bad.append(bad_policies(Q_d, env_death, env, dt))


    # Allocate outputs
    combined_size_map = np.zeros((n_dt, n_z), dtype=int)
    violation_map     = np.zeros((n_dt, n_z), dtype=int)  # ← new

    # Main loop
    # ——————————————————————————————————————————————————————————————
    for i, dt in enumerate(dt_vals):
        for j, zeta in enumerate(zeta_vals):
            pi_bad = lst_pi_bad[i]
            pi_svp = lst_pi_svp[j]

            count_total     = 0  # # states with k_svp + k_bad < |A|
            count_violations = 0  # # of those that are nonetheless inconsistent

            for s in range(env.nS):
                # number of actions selected by each policy
                k_svp = int(pi_svp[s].sum())
                k_bad = int(pi_bad[s].sum())

                # 1) record if this state satisfies the pigeon‐hole precondition
                # check if the number of selected actions of SVP+DeD is less than total actions
                if k_svp + k_bad < env.nA:
                    count_total += 1

                # 2) compute inconsistency flag
                is_inconsistent = False
                for a in range(env.nA):
                    if pi_svp[s, a] == 1 and pi_bad[s, a] == 1:
                        is_inconsistent = True
                        break

                # 3) if it met the pigeon‐hole condition but is still inconsistent, record a violation
                if (k_svp + k_bad < env.nA) and is_inconsistent:
                    count_violations += 1

            combined_size_map[i, j] = count_total
            violation_map    [i, j] = count_violations

    return {
        'zeta_vals':        zeta_vals,
        'dt_vals':          dt_vals,
        'combined_size_map': combined_size_map,
        'violation_map':     violation_map
    }


if __name__=="__main__":
    rs = np.random.RandomState(1234)
    env       = LifeGate(state_mode='tabular', rng=rs, death_drag=0.6, fixed_life=True)
    env_death = LifeGate(state_mode='tabular', rng=rs, death_drag=0.6, fixed_life=True)

    barrier_states  = [0,1,2,3,4,51,52,53,54]
    lifegate_states = [5,6,7]
    dead_states     = [8,9,19,29,39,49,59,69,79,89,99]
    dead_ends       = [45,46,47,48,55,56,57,58,65,66,67,68,
                       75,76,77,78,85,86,87,88,95,96,97,98]

    # train on a fine grid, ζ, dt ∈ [0,1] step=0.01
    zeta_vals = np.arange(0.01, 1.00, 0.01)
    dt_vals   = np.arange(0.01, 1.00, 0.01)

    results = train_search_pair(
      env, env_death, gamma=1,
      zeta_vals=zeta_vals, dt_vals=dt_vals,
      barrier_states=barrier_states,
      lifegate_states=lifegate_states,
      dead_states=dead_states,
      dead_ends=dead_ends,
      theta=1e-10, max_iter=1000
    )
    # save
    with open("results/trained_policies06.pkl","wb") as f:
        pickle.dump(results,f)
    print("Done training and saved to results/trained_policies06.pkl")


