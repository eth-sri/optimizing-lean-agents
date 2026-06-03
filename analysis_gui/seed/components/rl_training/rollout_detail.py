"""
Rollout detail tab for RL Training Analysis.

Shows detailed step-by-step analysis of individual rollouts.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict

from .utils import (
    load_rollouts_cached,
    load_single_rollout,
    get_rollouts_file,
    format_budget_display,
)
from typing import Optional


def render_rollout_detail_tab(run_dir: Path, run_metadata: Dict, target_budget: Optional[float]):
    """Render detailed rollout analysis with model predictions."""
    st.subheader(f"Rollout Detail Analysis - {format_budget_display(target_budget)} Budget")

    # Select round
    round_options = run_metadata.get('rounds', [0])
    selected_round = st.selectbox(
        "Select Round",
        options=round_options,
        index=len(round_options) - 1 if round_options else 0,
        key="rollout_detail_round"
    )

    # Load rollouts summary for selection UI (lightweight)
    round_dir = run_dir / f"round{selected_round}"
    rollouts_file = get_rollouts_file(round_dir)

    if rollouts_file is None:
        st.warning(f"No rollouts found for round {selected_round}")
        return

    # Load summary data for selection UI
    rollouts_summary = load_rollouts_cached(str(rollouts_file), target_budget)

    if not rollouts_summary:
        st.warning(f"No rollouts at {format_budget_display(target_budget)} budget")
        return

    # Get unique lambda values
    lambda_values = sorted(set(r.get('lambda', 0) for r in rollouts_summary))

    # Lambda selector
    if len(lambda_values) > 1:
        lambda_col1, lambda_col2 = st.columns([1, 3])
        with lambda_col1:
            selected_lambda = st.selectbox(
                "Select Lambda",
                options=lambda_values,
                format_func=lambda x: f"{float(x):.2e}",
                key="rollout_detail_lambda"
            )
        # Filter by lambda
        rollouts_summary = [r for r in rollouts_summary if r.get('lambda', 0) == selected_lambda]
    else:
        selected_lambda = lambda_values[0] if lambda_values else 0

    # Get unique problems
    problems = sorted(set(r.get('problem_id', r.get('origin_problem_id', 'unknown')) for r in rollouts_summary))

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_problem = st.selectbox(
            "Select Problem",
            options=problems,
            key="rollout_detail_problem"
        )

    # Filter rollouts for selected problem
    problem_rollouts = [r for r in rollouts_summary if r.get('problem_id', r.get('origin_problem_id', '')) == selected_problem]

    with col2:
        # Show rollout options with success indicator
        rollout_options = []
        for i, r in enumerate(problem_rollouts):
            success = "✅" if r.get('success', False) else "❌"
            seed = r.get('seed_idx', i)
            cost = r.get('final_cost', 0) / 1e6
            # Use num_steps from summary, fall back to history length for legacy
            steps = r.get('num_steps', len(r.get('history', [])))
            rollout_options.append(f"{success} Seed {seed} | {cost:.2f}M | {steps} steps")

        if not rollout_options:
            st.warning("No rollouts for this problem")
            return

        selected_rollout_idx = st.selectbox(
            "Select Rollout",
            options=range(len(rollout_options)),
            format_func=lambda i: rollout_options[i],
            key="rollout_detail_idx"
        )

    # Get the summary data for selected rollout (for metadata display)
    selected_summary = problem_rollouts[selected_rollout_idx]
    selected_seed = selected_summary.get('seed_idx', selected_rollout_idx)

    with col3:
        st.metric("Success", "✅ Yes" if selected_summary.get('success') else "❌ No")

    # Rollout summary (from summary data - always available)
    st.markdown("---")
    summary_cols = st.columns(5)
    with summary_cols[0]:
        st.metric("Final Cost", f"{selected_summary.get('final_cost', 0)/1e6:.2f}M")
    with summary_cols[1]:
        st.metric("Steps", selected_summary.get('num_steps', len(selected_summary.get('history', []))))
    with summary_cols[2]:
        st.metric("Lambda", f"{float(selected_summary.get('lambda', 0)):.2e}")
    with summary_cols[3]:
        st.metric("Solve Method", selected_summary.get('solve_method', 'N/A'))
    with summary_cols[4]:
        st.metric("Termination", selected_summary.get('termination_reason', 'N/A'))

    # Step-by-step analysis - load full rollout on demand
    st.markdown("---")
    st.markdown("### Step-by-Step Trajectory")

    # Check if we already have history (legacy format) or need to load individually
    if 'history' in selected_summary and selected_summary['history']:
        # Legacy format - history is already loaded
        selected_rollout = selected_summary
        history = selected_rollout.get('history', [])
    else:
        # New format - load individual rollout file on demand
        with st.spinner("Loading full rollout data..."):
            selected_rollout = load_single_rollout(
                str(round_dir),
                selected_problem,
                float(selected_lambda),
                target_budget,
                selected_seed
            )

        if selected_rollout is None:
            st.warning("Could not load full rollout data. Individual rollout file not found.")
            st.info("This may happen if using summary-only data. Try regenerating rollouts with the updated code.")
            return

        history = selected_rollout.get('history', [])

    if not history:
        st.info("No history data available")
        return

    # Check if predictions are available
    has_predictions = any('predictions' in step for step in history)

    if has_predictions:
        st.success("Model predictions available for this rollout")
    else:
        st.info("No model predictions logged (run with updated code to capture predictions)")

    # Build step data for display
    step_data = []
    for i, step in enumerate(history):
        action = step.get('action', {})
        result = step.get('result', {})
        state = step.get('state', {})
        predictions = step.get('predictions', {})
        chosen_details = predictions.get('chosen_action_details', {})

        # Compute avg_proof_length feature
        avg_proof_length = state.get('avg_proof_length', None)
        if avg_proof_length is None:
            avg_full_8b = state.get('avg_proof_length_full_8b', 0)
            avg_full_32b = state.get('avg_proof_length_full_32b', 0)
            avg_attempt_8b = state.get('avg_proof_length_attempt_8b', 0)
            avg_attempt_32b = state.get('avg_proof_length_attempt_32b', 0)
            num_full_8b = state.get('num_full_proof_8b_used', 0)
            num_full_32b = state.get('num_full_proof_32b_used', 0)
            num_attempt_8b = state.get('num_attempt_8b_on_lemma', 0)
            num_attempt_32b = state.get('num_attempt_32b_on_lemma', 0)
            total_count = num_full_8b + num_full_32b + num_attempt_8b + num_attempt_32b
            if total_count > 0:
                avg_proof_length = (
                    avg_full_8b * num_full_8b + avg_full_32b * num_full_32b +
                    avg_attempt_8b * num_attempt_8b + avg_attempt_32b * num_attempt_32b
                ) / total_count
            else:
                non_zero = [v for v in [avg_full_8b, avg_full_32b, avg_attempt_8b, avg_attempt_32b] if v > 0]
                avg_proof_length = sum(non_zero) / len(non_zero) if non_zero else 0

        row = {
            'Step': i + 1,
            'Action': action.get('action_type', 'N/A'),
            'n_attempts': state.get('num_attempts_made', 0),
            'avg_pf_len': f"{avg_proof_length:.0f}",
            'pf_len': result.get('proof_length', 0),
            'Success': '✅' if result.get('success', False) else '❌',
            'Step Cost': f"{step.get('step_cost', 0)/1e6:.3f}M",
            'Total Cost': f"{step.get('cost_so_far', 0)/1e6:.3f}M",
        }

        if chosen_details:
            row['P(Succ)'] = f"{chosen_details.get('predicted_success_prob', 0):.3f}"
            row['Pred Cost'] = f"{chosen_details.get('predicted_cost', 0)/1e6:.3f}M"
            row['Q'] = f"{chosen_details.get('q_value', 0):.4f}"

        step_data.append(row)

    step_df = pd.DataFrame(step_data)
    st.dataframe(step_df, use_container_width=True, hide_index=True)

    # Detailed step viewer
    st.markdown("---")
    st.markdown("### Step Details")

    step_options = []
    for i, s in enumerate(history):
        act = s.get('action', {}).get('action_type', 'N/A')
        success = "✅" if s.get('result', {}).get('success', False) else "❌"
        lemma_id = s.get('state', {}).get('lemma_id', -1)
        step_options.append(f"Step {i+1}: {act} (L{lemma_id}) {success}")

    selected_step_idx = st.selectbox(
        "Select Step to Inspect",
        options=range(len(history)),
        format_func=lambda i: step_options[i],
        key="rollout_step_select"
    )

    step = history[selected_step_idx]
    action = step.get('action', {})
    result = step.get('result', {})
    state = step.get('state', {})
    predictions = step.get('predictions', {})

    detail_cols = st.columns(3)

    with detail_cols[0]:
        st.markdown("**Model Input Features**")

        # Show input features from predictions if available (these are the actual model inputs)
        input_features = predictions.get('input_features', {})
        if input_features:
            for feat_name, feat_value in input_features.items():
                # Format cost features in millions, others as-is
                if 'cost' in feat_name.lower():
                    st.write(f"  **{feat_name}**: `{feat_value/1e6:.4f}M` ({feat_value:.0f})")
                elif isinstance(feat_value, float):
                    st.write(f"  **{feat_name}**: `{feat_value:.4f}`")
                else:
                    st.write(f"  **{feat_name}**: `{feat_value}`")
        else:
            # Fallback to computing from state (for old rollouts without input_features)
            num_attempts = state.get('num_attempts_made', 0)
            st.write(f"  **num_attempts_made**: `{num_attempts}`")

            # Show key cost features from state
            num_full_8b = state.get('num_full_proof_8b_used', 0)
            num_full_32b = state.get('num_full_proof_32b_used', 0)
            cost_8b = state.get('cost_so_far_full_proof_8b', 0)
            cost_32b = state.get('cost_so_far_full_proof_32b', 0)
            avg_cost_8b = cost_8b / max(num_full_8b, 1) if num_full_8b > 0 else 0
            avg_cost_32b = cost_32b / max(num_full_32b, 1) if num_full_32b > 0 else 0

            st.write(f"  **avg_cost_full_8b**: `{avg_cost_8b/1e6:.4f}M`")
            st.write(f"  **avg_cost_full_32b**: `{avg_cost_32b/1e6:.4f}M`")
            st.write(f"  **num_full_proof_8b_used**: `{num_full_8b}`")
            st.write(f"  **num_full_proof_32b_used**: `{num_full_32b}`")

        # Show context info in smaller text
        with st.expander("Raw State", expanded=False):
            st.write(f"  lemma_id: `{state.get('lemma_id', -1)}`")
            st.write(f"  breakdown_id: `{state.get('breakdown_id', -1)}`")
            st.write(f"  proof_length: `{state.get('proof_length', 0)}`")
            st.write(f"  num_errors: `{state.get('num_errors', 0)}`")
            st.write(f"  cost_so_far_full_proof_8b: `{state.get('cost_so_far_full_proof_8b', 0):.0f}`")
            st.write(f"  cost_so_far_full_proof_32b: `{state.get('cost_so_far_full_proof_32b', 0):.0f}`")

    with detail_cols[1]:
        st.markdown("**Action & Result**")
        st.write(f"  Action: `{action.get('action_type', 'N/A')}`")
        st.write(f"  Attempt Index: `{action.get('attempt_index', 0)}`")
        st.write(f"  Success: `{result.get('success', False)}`")
        st.write(f"  Result Proof Length: `{result.get('proof_length', 0)}`")
        st.write(f"  Result Errors: `{result.get('num_errors', 0)}`")
        st.write(f"  Step Cost: `{step.get('step_cost', 0)/1e6:.4f}M`")

    with detail_cols[2]:
        st.markdown("**Chosen Action Predictions**")
        if predictions:
            chosen = predictions.get('chosen_action_details', {})
            st.write(f"  P(Success): `{chosen.get('predicted_success_prob', 'N/A')}`")
            st.write(f"  Pred Cost: `{chosen.get('predicted_cost', 0)/1e6:.4f}M`")
            st.write(f"  Q-Value: `{chosen.get('q_value', 'N/A')}`")
        else:
            st.write("  No predictions available")

    # Show all action scores as a table below the columns
    if predictions:
        all_scores = predictions.get('all_action_scores', [])
        if all_scores:
            st.markdown("---")
            st.markdown("**All Action Predictions** (Q = P(success) - λ × cost)")

            lambda_val = float(selected_rollout.get('lambda', 0))
            st.caption(f"Lambda: {lambda_val:.2e}")

            chosen_action = action.get('action_type', '')
            action_table_data = []
            for score_info in all_scores:
                action_name = score_info.get('action', 'N/A')
                q_val = score_info.get('q_value', 0)
                p_succ = score_info.get('success_prob', 0)
                p_cost = score_info.get('predicted_cost', 0)
                is_chosen = action_name == chosen_action
                arrow = "→ " if is_chosen else ""

                action_table_data.append({
                    'Action': f"{arrow}{action_name}",
                    'P(Success)': f"{p_succ:.4f}",
                    'Pred Cost': f"{p_cost/1e6:.4f}M",
                    'λ × Cost': f"{lambda_val * p_cost:.4f}",
                    'Q (P - λC)': f"{q_val:.4f}",
                })

            action_table_df = pd.DataFrame(action_table_data)
            st.dataframe(action_table_df, use_container_width=True, hide_index=True)

    # Visualization of trajectory
    if has_predictions and len(history) > 1:
        st.markdown("---")
        st.markdown("### Trajectory Visualization")

        # Extract data for plotting
        steps_x = list(range(1, len(history) + 1))
        cumulative_costs = [step.get('cost_so_far', 0) / 1e6 for step in history]

        # Extract metrics for all actions at each step
        all_action_q_values = {}  # action_name -> list of q values per step
        all_action_success_probs = {}  # action_name -> list of success probs per step
        all_action_pred_costs = {}  # action_name -> list of predicted costs per step
        chosen_actions = []

        for step in history:
            preds = step.get('predictions', {})
            chosen_actions.append(step.get('action', {}).get('action_type', ''))

            # Get all action scores at this step
            all_scores = preds.get('all_action_scores', [])
            step_q_by_action = {s.get('action'): s.get('q_value') for s in all_scores}
            step_prob_by_action = {s.get('action'): s.get('success_prob') for s in all_scores}
            step_cost_by_action = {s.get('action'): s.get('predicted_cost') for s in all_scores}

            # Update Q values
            for action_name, q_val in step_q_by_action.items():
                if action_name not in all_action_q_values:
                    all_action_q_values[action_name] = [None] * (len(chosen_actions) - 1)
                all_action_q_values[action_name].append(q_val)

            # Update success probs
            for action_name, prob in step_prob_by_action.items():
                if action_name not in all_action_success_probs:
                    all_action_success_probs[action_name] = [None] * (len(chosen_actions) - 1)
                all_action_success_probs[action_name].append(prob)

            # Update predicted costs
            for action_name, cost in step_cost_by_action.items():
                if action_name not in all_action_pred_costs:
                    all_action_pred_costs[action_name] = [None] * (len(chosen_actions) - 1)
                # Convert to millions for display
                all_action_pred_costs[action_name].append(cost / 1e6 if cost is not None else None)

            # Fill None for actions not present in this step
            for action_name in all_action_q_values:
                if len(all_action_q_values[action_name]) < len(chosen_actions):
                    all_action_q_values[action_name].append(None)
            for action_name in all_action_success_probs:
                if len(all_action_success_probs[action_name]) < len(chosen_actions):
                    all_action_success_probs[action_name].append(None)
            for action_name in all_action_pred_costs:
                if len(all_action_pred_costs[action_name]) < len(chosen_actions):
                    all_action_pred_costs[action_name].append(None)

        action_colors = {
            'FULL_PROOF_8B': 'blue',
            'FULL_PROOF_32B': 'red',
            'ATTEMPT_8B': 'lightblue',
            'ATTEMPT_32B': 'orange',
            'CREATE_BREAKDOWN': 'green',
            'CORRECTION_32B': 'purple',
            'TERMINATE': 'gray',
        }

        # Create subplot with 4 panels
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cumulative Cost (Actual)',
                'P(Success) by Action',
                'Predicted Cost by Action',
                'Q-Value by Action'
            ],
            specs=[[{}, {}], [{}, {}]]
        )

        # 1. Cumulative cost over time (actual)
        fig.add_trace(
            go.Scatter(x=steps_x, y=cumulative_costs, mode='lines+markers', name='Cumulative Cost',
                      line=dict(color='black'), showlegend=False),
            row=1, col=1
        )

        # 2. Success prob by action
        for action_name, probs in all_action_success_probs.items():
            if any(p is not None for p in probs):
                fig.add_trace(
                    go.Scatter(
                        x=steps_x,
                        y=probs,
                        mode='lines+markers',
                        name=action_name,
                        line=dict(color=action_colors.get(action_name, 'gray')),
                        legendgroup=action_name,
                    ),
                    row=1, col=2
                )

        # 3. Predicted remaining cost by action
        for action_name, costs in all_action_pred_costs.items():
            if any(c is not None for c in costs):
                fig.add_trace(
                    go.Scatter(
                        x=steps_x,
                        y=costs,
                        mode='lines+markers',
                        name=action_name,
                        line=dict(color=action_colors.get(action_name, 'gray')),
                        legendgroup=action_name,
                        showlegend=False,  # Already shown from success prob
                    ),
                    row=2, col=1
                )

        # 4. Q-values by action
        for action_name, q_values in all_action_q_values.items():
            if any(q is not None for q in q_values):
                fig.add_trace(
                    go.Scatter(
                        x=steps_x,
                        y=q_values,
                        mode='lines+markers',
                        name=action_name,
                        line=dict(color=action_colors.get(action_name, 'gray')),
                        legendgroup=action_name,
                        showlegend=False,  # Already shown from success prob
                    ),
                    row=2, col=2
                )

        # Update axis labels
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=2)
        fig.update_yaxes(title_text="Cost (M)", row=1, col=1)
        fig.update_yaxes(title_text="P(Success)", row=1, col=2)
        fig.update_yaxes(title_text="Pred Cost (M)", row=2, col=1)
        fig.update_yaxes(title_text="Q-Value", row=2, col=2)

        fig.update_layout(height=600, showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
