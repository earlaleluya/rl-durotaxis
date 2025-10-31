# Reward Equations (Formal Specification)

This document defines the reward components used by `DurotaxisEnv` with consistent notation and explicit equations, aligned to the current implementation in `durotaxis_env.py`.

## Notation

- Time: step index t, with state s_t and next state s_{t+1}
- Node set at time t: N_t = {1, …, n_t}, n_t = |N_t|
- Substrate dimensions: width W, height H, goal x-position G = W − 1
- Node i position: (x_i^t, y_i^t)
- Substrate intensity at node i: I_i^t
- Boundary flag at node i: b_i^t ∈ {0,1}
- To-delete flag at node i (evaluated at t): d_i^t ∈ {0,1}
- Action set taken at t: A_t, with |A_t| actions
- Centroid x-coordinate: C_x(s_t)
- Indicator function: 𝟙[·]
- Discount factor for PBRS: γ ∈ (0,1]
- All scalar coefficients are pulled from config; we denote them by their names below

Total per-step reward R_t is composed from graph-level, node-level, survival and milestone components, with mode-dependent overrides.

## Graph-level base components

1) Connectivity/growth component

- If n_{t+1} < 2:
  $$R_{conn} = -\text{connectivity\_penalty}$$
- Else if n_{t+1} > N_c (max_critical_nodes):
  $$R_{grow} = -\text{growth\_penalty}\,\Big(1 + \frac{n_{t+1} - N_c}{N_c}\Big)$$
- Else:
  $$R_{survive} = +\text{survival\_reward}$$

Only one of the above three applies per step.

2) Action count reward

$$R_{act} = |A_t| \cdot \text{action\_reward}$$

3) Centroid movement reward (+ PBRS)

Base term:
$$R_{centroid}^{base} = \text{centroid\_movement\_reward} \cdot \big(C_x(s_{t+1}) - C_x(s_t)\big)$$

Optional PBRS shaping (if enabled):
- Potential: $$\Phi_{cen}(s) = -\text{phi\_distance\_scale}\cdot\big(G - C_x(s)\big)$$
- Shaping: $$F_{cen}(s_t,s_{t+1}) = \gamma\,\Phi_{cen}(s_{t+1}) - \Phi_{cen}(s_t)$$
- Applied as: $$R_{centroid} = R_{centroid}^{base} + \text{shaping\_coeff}_{cen}\cdot F_{cen}(s_t,s_{t+1})$$

4) Spawn reward (durotaxis-based) with boundary checks

Let J_{new} be indices of nodes spawned at t+1 (new_node flag = 1). For each j ∈ J_{new}:
- Find parent p in s_t minimizing Euclidean distance to j
- Intensity difference: ΔI_j = I_j^{t+1} − I_p^{t}
- Intensity term:
  $$r^{intensity}_j = \begin{cases}
  +\text{spawn\_success\_reward}, & \Delta I_j \ge \Delta I_{\min}~(\text{delta\_intensity}) \\
  -\text{spawn\_failure\_penalty}, & \text{otherwise}
  \end{cases}$$
- Boundary penalty: define d_j = \min\{ y_j^{t+1}/H, (H - y_j^{t+1})/H \}
  $$r^{bound}_j = \begin{cases}
  -\text{spawn\_in\_danger\_zone\_penalty}, & d_j < \text{danger\_zone\_threshold} \\
  -\text{spawn\_near\_boundary\_penalty}, & \text{danger\_zone\_threshold} \le d_j < \text{edge\_zone\_threshold} \\
  0, & \text{otherwise}
  \end{cases}$$
Total spawn reward:
$$R_{spawn} = \sum_{j\in J_{new}} \big( r^{intensity}_j + r^{bound}_j \big)$$

5) Delete compliance reward (+ PBRS)

Let P_t be the set of persistent IDs at t. For each i ∈ P_t with delete flag d_i^t:
- Let E_{t+1}(i) be indicator that i exists at t+1
- Casework:
  $$r^{del}_i = \begin{cases}
  +\text{proper\_deletion}, & d_i^t=1 \wedge E_{t+1}(i)=0 \\
  -\text{persistence\_penalty}, & d_i^t=1 \wedge E_{t+1}(i)=1 \\
  -\text{improper\_deletion\_penalty}, & d_i^t=0 \wedge E_{t+1}(i)=0 \\
  +\text{proper\_deletion}, & d_i^t=0 \wedge E_{t+1}(i)=1
  \end{cases}$$
Base delete reward:
$$R_{delete}^{base} = \sum_{i\in P_t} r^{del}_i$$

Optional PBRS shaping (if enabled):
- Potential with counts over s: pending_marked(s) = \sum_i 𝟙[d_i=1], safe_unmarked(s) = \sum_i 𝟙[d_i=0]
  $$\Phi_{del}(s) = -w_{pend}\,\mathrm{pending\_marked}(s) + w_{safe}\,\mathrm{safe\_unmarked}(s)$$
- Shaping: $$F_{del}(s_t,s_{t+1}) = \gamma\,\Phi_{del}(s_{t+1}) - \Phi_{del}(s_t)$$
- Applied as: $$R_{delete} = R_{delete}^{base} + \text{shaping\_coeff}_{del}\cdot F_{del}(s_t,s_{t+1})$$

6) Deletion efficiency (tidiness)

Let D = P_t \setminus P_{t+1} be deleted IDs. For each pid ∈ D with age a(pid) and stagnation count z(pid):
$$r^{eff}_{pid} = \max\{0,\,(a(pid)-50)\}\cdot \rho_{age} + \max\{0,\,(z(pid)-20)\}\cdot \rho_{stag}$$
Total:
$$R_{efficiency} = \sum_{pid\in D} r^{eff}_{pid}$$

7) Edge direction reward

For each directed edge (u→v) at t+1 with positions x_u^{t+1}, x_v^{t+1} and threshold ε≈0.01:
$$r^{edge}_{u,v} = \begin{cases}
+\text{rightward\_bonus}, & x_v^{t+1} - x_u^{t+1} > \varepsilon \\
-\text{leftward\_penalty}, & x_v^{t+1} - x_u^{t+1} < -\varepsilon \\
0, & \text{otherwise}
\end{cases}$$
Total:
$$R_{edge} = \sum_{(u\to v)} r^{edge}_{u,v}$$

## Node-level components (vectorized)

Per-node reward r_i aggregated as a sum of terms; total node reward is \(R_{node} = \sum_{i\in N_{t+1}} r_i\).

1) Movement term (requires previous positions)

Let Δx_i = x_i^{t+1} − x_i^{t}:
$$r^{move}_i = \begin{cases}
\Delta x_i\cdot \text{movement\_reward}, & \Delta x_i > 0 \\
\Delta x_i\cdot \text{leftward\_penalty}, & \Delta x_i \le 0
\end{cases}$$

2) Substrate intensity term

$$r^{sub}_i = I_i^{t+1}\cdot \text{substrate\_reward}$$

3) Boundary (convex hull/frontier) bonus

$$r^{bnd}_i = \text{boundary\_bonus}\cdot 𝟙[b_i^{t+1} > 0.5]$$

4) Left-edge penalty (avoid spawn near left side)

$$r^{left}_i = -\text{left\_edge\_penalty}\cdot 𝟙[x_i^{t+1} < 0.1\,W]$$

5) Top/bottom proximity penalties (progressive)

Let \(d_i = \min\{ y_i^{t+1}/H,\, (H - y_i^{t+1})/H \}\). With thresholds:
- edge_zone_threshold = τ_e, danger_zone_threshold = τ_d, critical_zone_threshold = τ_c

Piecewise penalty:
$$r^{tb}_i = \begin{cases}
-\text{critical\_zone\_penalty}, & d_i < \tau_c \\
-\text{danger\_zone\_penalty}, & \tau_c \le d_i < \tau_d \\
-\text{edge\_position\_penalty}, & \tau_d \le d_i < \tau_e \\
0, & d_i \ge \tau_e
\end{cases}$$

6) Safe-center bonus

Let \(c_i = \frac{|y_i^{t+1} - H/2|}{H/2}\). With safe_center_range = ρ_c:
$$r^{safe}_i = \text{safe\_center\_bonus}\cdot 𝟙[c_i < \rho_c]$$

7) Intensity vs average (historical comparison)

Let \(\bar I^{t+1} = \frac{1}{n_{t+1}}\sum_{i} I_i^{t+1}\). For nodes that existed in the dequeued topology snapshot:
$$r^{avg}_i = \begin{cases}
-\text{intensity\_penalty}, & I_i^{t+1} < \bar I^{t+1} \\
+\text{intensity\_bonus}, & I_i^{t+1} \ge \bar I^{t+1}
\end{cases}$$

Total per-node reward:
$$r_i = r^{move}_i + r^{sub}_i + r^{bnd}_i + r^{left}_i + r^{tb}_i + r^{safe}_i + r^{avg}_i$$

## Survival and milestone rewards

1) Survival reward (time-based)

If enabled, an additional per-step survival term may be added by configuration:
$$R_{survival(t)} = f_{survival}(t)$$

2) Milestone rewards (distance thresholds)

Let x_{max}^{t+1} = \max_i x_i^{t+1} and progress p = 100\cdot x_{max}^{t+1}/W. With thresholds at {25,50,75,90}% each granting a one-time reward M_{25}, M_{50}, M_{75}, M_{90} respectively when first crossed in the episode:
$$R_{milestone} = \sum_{m\in\{25,50,75,90\}} M_m\cdot 𝟙[p \ge m \wedge m \text{ not yet reached}]$$

Note: Milestone rewards are disabled inside centroid_distance_only_mode.

## Mode-dependent total reward composition

Let the default combined scalar before termination adjustments be:
$$R_{base} = R_{graph} + R_{node} + R_{survival(t)} + R_{milestone}$$
where
$$R_{graph} = R_{conn/grow/survive} + R_{act} + R_{centroid} + R_{spawn} + R_{delete} + R_{efficiency} + R_{edge}$$

Modes:

1) Normal mode (both special modes disabled)
- Use full composition: \(R_{total} = R_{base}\)
- On termination, add termination reward directly: \(R_{total} \leftarrow R_{total} + R_{term}\)

2) Simple delete-only mode (simple_delete_only_mode = True)
- Step reward focuses on deletion only:
  $$R_{total} = R_{delete}$$
- On termination (if include_termination_rewards = True), the environment sets:
  $$R_{total} = R_{graph} + R_{term}$$
  where \(R_{graph}\) is as computed that step (kept for logging consistency)

3) Centroid-distance-only mode (centroid_distance_only_mode = True)
- Step reward focuses on distance signal:
  $$R_{total} = R_{distance\,signal}$$
  where the implementation uses delta-based shaping when enabled:
  $$R_{distance\,signal} = \text{dm\_dist\_scale}\cdot \frac{C_x(s_{t+1}) - C_x(s_t)}{G} \quad \text{(if delta enabled)}$$
  else static penalty: \(R = -(G - C_x(s_{t+1}))/G\)
- On termination (if include_termination_rewards = True), the environment adds scaled+clipped termination:
  $$R_{total} \leftarrow R_{total} + \mathrm{clip}\big(\text{dm\_term\_scale}\cdot R_{term},\, -\text{dm\_term\_clip\_val},\, \text{dm\_term\_clip\_val}\big)$$

4) Combined mode (both special modes True)
- Step reward combines distance signal and delete reward:
  $$R_{total} = R_{distance\,signal} + R_{delete}$$
- On termination (if include_termination_rewards = True), same scaled+clipped termination as above is added to \(R_{total}\).

## Termination reward/penalty

At termination, \(R_{term}\) is determined by outcome:
- Success: \(+\text{success\_reward}\)
- Out-of-bounds: \(\text{out\_of\_bounds\_penalty}\)
- No nodes: \(\text{no\_nodes\_penalty}\)
- Leftward drift: \(\text{leftward\_drift\_penalty}\)
- Timeout: \(\text{timeout\_penalty}\)
- Critical nodes: \(\text{critical\_nodes\_penalty}\)

As discussed, termination rewards are not PBRS-shaped; PBRS applies only to dense step rewards (centroid movement and delete compliance).

## Summary

- All dense components are explicitly defined by the equations above and correspond to the implementation.
- Optional PBRS is applied to centroid movement and delete compliance via shaping terms \(\gamma\Phi(s') - \Phi(s)\), preserving optimal policy.
- Mode switches (simple delete only, centroid distance only, combined) modify how \(R_{total}\) is composed each step and how termination is injected.
