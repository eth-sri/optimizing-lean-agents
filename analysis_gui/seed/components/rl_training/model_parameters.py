"""
Model parameters tab for RL Training Analysis.

Shows model coefficients and feature importance across training rounds.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional

from .utils import load_model_parameters


def render_model_parameters_tab(run_dir: Path, run_metadata: Dict):
    """Render model parameters analysis showing coefficients and feature importance."""
    st.subheader("Model Parameters Analysis")

    rounds = run_metadata.get('rounds', [])

    # Check for initial tree rollout model (saved before any rounds)
    tree_model_dir = run_dir / "tree_rollout_model"
    has_initial_model = tree_model_dir.exists()

    # Check which rounds have trained models
    rounds_with_models = []
    for round_id in rounds:
        model_dir = run_dir / f"round{round_id}" / "trained_models"
        if model_dir.exists():
            rounds_with_models.append(round_id)

    if not rounds_with_models and not has_initial_model:
        st.warning("No trained models found in any round.")
        st.info("Models are saved to `roundX/trained_models/` after training.")
        return

    # Build round options including initial model
    round_options = []
    if has_initial_model:
        round_options.append("initial")
    round_options.extend(rounds_with_models)

    st.caption(f"Found models: {'initial (tree rollout), ' if has_initial_model else ''}{', '.join(f'round {r}' for r in rounds_with_models) if rounds_with_models else 'none'}")

    # Round selector
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_round = st.selectbox(
            "Select Round",
            options=round_options,
            format_func=lambda x: "Initial (tree rollout)" if x == "initial" else f"Round {x}",
            index=len(round_options) - 1,
            key="model_params_round"
        )

    # Load model parameters
    if selected_round == "initial":
        model_dir = run_dir / "tree_rollout_model"
    else:
        model_dir = run_dir / f"round{selected_round}" / "trained_models"
    model_data = load_model_parameters(model_dir)

    if not model_data:
        st.error(f"Failed to load models from {model_dir}")
        return

    # Sub-tabs for success vs cost models
    model_tab1, model_tab2, model_tab3 = st.tabs([
        "Success Model (LogisticRegression)",
        "Cost Model (LinearRegression)",
        "Cross-Round Comparison"
    ])

    # Format round label for display
    round_label = "Initial (tree rollout)" if selected_round == "initial" else f"Round {selected_round}"

    with model_tab1:
        render_success_model_params(model_data.get('success'), round_label)

    with model_tab2:
        render_cost_model_params(model_data.get('cost'), round_label)

    with model_tab3:
        render_cross_round_comparison(run_dir, rounds_with_models)


def render_success_model_params(success_data: Optional[Dict], round_label: str):
    """Render success model parameters."""
    if not success_data:
        st.info("No success model data available.")
        return

    st.markdown(f"### Success Model Parameters ({round_label})")
    st.markdown("*Logistic regression predicting P(rollout succeeds | state, action)*")

    models = success_data.get('models', {})
    scaler = success_data.get('scaler', None)  # Single shared scaler
    feature_names = success_data.get('feature_names', None)

    # Check for model-specific feature lists (tree rollout models)
    features_8b = success_data.get('features_8b', None)
    features_32b = success_data.get('features_32b', None)
    has_model_specific_features = features_8b or features_32b

    if not models:
        st.warning("No models found in success data.")
        return

    # Infer feature names if not stored
    if not feature_names and scaler:
        num_features = scaler.n_features_in_
        feature_names = [f"feature_{i}" for i in range(num_features)]

    # Show feature info
    if has_model_specific_features:
        st.markdown(f"**8B Features:** {features_8b if features_8b else 'Not stored'}")
        st.markdown(f"**32B Features:** {features_32b if features_32b else 'Not stored'}")
    else:
        st.markdown(f"**Features ({len(feature_names) if feature_names else 'unknown'}):** {feature_names if feature_names else 'Not stored'}")

    # Build coefficient table
    coef_data = []
    for action_type, model in models.items():
        action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)

        # Get model-specific feature names if available
        if has_model_specific_features:
            if '8B' in action_name:
                model_features = features_8b or feature_names
            else:
                model_features = features_32b or feature_names
        else:
            model_features = feature_names

        if hasattr(model, 'coef_'):
            coefs = model.coef_.flatten()
            intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_

            row = {'Action': action_name, 'Intercept': intercept}
            for i, coef in enumerate(coefs):
                feat_name = model_features[i] if model_features and i < len(model_features) else f"feat_{i}"
                row[feat_name] = coef
            coef_data.append(row)

    if coef_data:
        st.markdown("#### Coefficients by Action Type")
        coef_df = pd.DataFrame(coef_data)

        # Reorder columns: Action, Intercept, then features
        cols = ['Action', 'Intercept'] + [c for c in coef_df.columns if c not in ['Action', 'Intercept']]
        coef_df = coef_df[cols]

        # Format numeric columns
        numeric_cols = [c for c in coef_df.columns if c != 'Action']
        for col in numeric_cols:
            coef_df[col] = coef_df[col].apply(lambda x: f"{x:.4f}")

        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Visualization: coefficient heatmap
        st.markdown("#### Coefficient Heatmap")
        _render_coefficient_heatmap(coef_data, feature_names, "Success Model")

    # Scaler parameters
    if scaler and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        st.markdown("#### Scaler Parameters (StandardScaler)")
        features_to_scale = success_data.get('features_to_scale', [])
        if features_to_scale:
            st.caption(f"Only these features are scaled: {features_to_scale}")
        scaler_data = []
        for i, (mean, scale) in enumerate(zip(scaler.mean_, scaler.scale_)):
            feat_name = features_to_scale[i] if features_to_scale and i < len(features_to_scale) else f"feat_{i}"
            scaler_data.append({
                'Feature': feat_name,
                'Mean': f"{mean:.4f}",
                'Scale': f"{scale:.4f}",
            })

        scaler_df = pd.DataFrame(scaler_data)
        st.dataframe(scaler_df, use_container_width=True, hide_index=True)

    # Calibration parameters (for tree rollout models)
    calibration_8b = success_data.get('calibration_8b')
    calibration_32b = success_data.get('calibration_32b')
    if calibration_8b or calibration_32b:
        st.markdown("#### Platt Scaling Calibration Parameters")
        st.caption("Calibration: P(success) = sigmoid(a * logit + b)")
        calib_data = []
        if calibration_8b:
            # Handle both dict format {'a': val, 'b': val} and tuple/list format (a, b)
            if isinstance(calibration_8b, dict):
                a, b = calibration_8b['a'], calibration_8b['b']
            else:
                a, b = calibration_8b
            calib_data.append({'Model': 'FULL_PROOF_8B', 'a (slope)': f"{float(a):.4f}", 'b (intercept)': f"{float(b):.4f}"})
        if calibration_32b:
            if isinstance(calibration_32b, dict):
                a, b = calibration_32b['a'], calibration_32b['b']
            else:
                a, b = calibration_32b
            calib_data.append({'Model': 'FULL_PROOF_32B', 'a (slope)': f"{float(a):.4f}", 'b (intercept)': f"{float(b):.4f}"})
        if calib_data:
            calib_df = pd.DataFrame(calib_data)
            st.dataframe(calib_df, use_container_width=True, hide_index=True)


def render_cost_model_params(cost_data: Optional[Dict], round_label: str):
    """Render cost model parameters."""
    if not cost_data:
        st.info("No cost model data available.")
        return

    st.markdown(f"### Cost Model Parameters ({round_label})")
    st.markdown("*Linear regression predicting remaining cost (action cost + future cost)*")

    models = cost_data.get('models', {})
    scaler = cost_data.get('scaler', None)  # Single shared scaler
    feature_names = cost_data.get('feature_names', None)

    if not models:
        st.warning("No models found in cost data.")
        return

    # Infer feature names if not stored
    if not feature_names and scaler:
        num_features = scaler.n_features_in_
        feature_names = [f"feature_{i}" for i in range(num_features)]

    st.markdown(f"**Features ({len(feature_names) if feature_names else 'unknown'}):** {feature_names if feature_names else 'Not stored'}")

    # Build coefficient table
    coef_data = []
    for action_type, model in models.items():
        action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)

        if hasattr(model, 'coef_'):
            coefs = model.coef_.flatten()
            intercept = float(model.intercept_)

            row = {'Action': action_name, 'Intercept': intercept}
            for i, coef in enumerate(coefs):
                feat_name = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i}"
                row[feat_name] = coef
            coef_data.append(row)

    if coef_data:
        st.markdown("#### Coefficients by Action Type")
        coef_df = pd.DataFrame(coef_data)

        # Reorder columns
        cols = ['Action', 'Intercept'] + [c for c in coef_df.columns if c not in ['Action', 'Intercept']]
        coef_df = coef_df[cols]

        # Format: intercept in millions for cost model
        display_df = coef_df.copy()
        display_df['Intercept'] = display_df['Intercept'].apply(lambda x: f"{float(x)/1e6:.4f}M")
        for col in [c for c in display_df.columns if c not in ['Action', 'Intercept']]:
            display_df[col] = display_df[col].apply(lambda x: f"{float(x):.4f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Visualization
        st.markdown("#### Coefficient Heatmap")
        _render_coefficient_heatmap(coef_data, feature_names, "Cost Model")

    # Shared scaler parameters
    if scaler and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        st.markdown("#### Shared Scaler Parameters (StandardScaler)")
        scaler_data = []
        for i, (mean, scale) in enumerate(zip(scaler.mean_, scaler.scale_)):
            feat_name = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i}"
            scaler_data.append({
                'Feature': feat_name,
                'Mean': f"{mean:.4f}",
                'Scale': f"{scale:.4f}",
            })

        scaler_df = pd.DataFrame(scaler_data)
        st.dataframe(scaler_df, use_container_width=True, hide_index=True)


def _render_coefficient_heatmap(coef_data: List[Dict], feature_names: Optional[List[str]], title: str):
    """Render a heatmap of coefficients across action types."""
    if not coef_data:
        return

    # Build matrix for heatmap
    actions = [row['Action'] for row in coef_data]
    features = ['Intercept'] + (feature_names if feature_names else [])

    # Ensure we have feature columns in data
    if not feature_names:
        # Infer from first row
        first_row = coef_data[0]
        features = ['Intercept'] + [k for k in first_row.keys() if k not in ['Action', 'Intercept']]

    z = []
    for row in coef_data:
        z_row = [row.get('Intercept', 0)]
        for feat in features[1:]:  # Skip 'Intercept' which we already added
            z_row.append(row.get(feat, 0))
        z.append(z_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=features,
        y=actions,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.3f}" for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Action: %{y}<br>Feature: %{x}<br>Coefficient: %{z:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"{title} Coefficients",
        xaxis_title="Feature",
        yaxis_title="Action Type",
        height=max(300, len(actions) * 40 + 100),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cross_round_comparison(run_dir: Path, rounds_with_models: List[int]):
    """Render comparison of model parameters across rounds."""
    st.markdown("### Cross-Round Model Evolution")

    if len(rounds_with_models) < 2:
        st.info("Need at least 2 rounds with models for comparison.")
        return

    # Model type selector
    model_type = st.radio(
        "Model Type",
        options=["success", "cost"],
        format_func=lambda x: "Success Model" if x == "success" else "Cost Model",
        horizontal=True,
        key="cross_round_model_type"
    )

    # Load all models
    all_models_data = {}
    for round_id in rounds_with_models:
        model_dir = run_dir / f"round{round_id}" / "trained_models"
        model_data = load_model_parameters(model_dir)
        if model_data and model_type in model_data:
            all_models_data[round_id] = model_data[model_type]

    if len(all_models_data) < 2:
        st.warning(f"Not enough {model_type} models found for comparison.")
        return

    # Get action types from first model
    first_round = list(all_models_data.keys())[0]
    first_models = all_models_data[first_round].get('models', {})
    action_types = list(first_models.keys())

    if not action_types:
        st.warning("No action types found in models.")
        return

    # Action type selector
    action_options = [at.name if hasattr(at, 'name') else str(at) for at in action_types]
    selected_action_name = st.selectbox(
        "Select Action Type",
        options=action_options,
        key="cross_round_action"
    )

    # Find matching action type
    selected_action = None
    for at in action_types:
        at_name = at.name if hasattr(at, 'name') else str(at)
        if at_name == selected_action_name:
            selected_action = at
            break

    if selected_action is None:
        st.error("Could not find selected action type.")
        return

    # Get feature names
    feature_names = all_models_data[first_round].get('feature_names', None)
    if not feature_names:
        scaler = all_models_data[first_round].get('scaler', None)
        if scaler and hasattr(scaler, 'n_features_in_'):
            feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]

    # Build evolution data
    evolution_data = []
    for round_id in sorted(all_models_data.keys()):
        models = all_models_data[round_id].get('models', {})
        if selected_action in models:
            model = models[selected_action]
            if hasattr(model, 'coef_'):
                coefs = model.coef_.flatten()
                intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_

                row = {'Round': round_id, 'Intercept': intercept}
                for i, coef in enumerate(coefs):
                    feat_name = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i}"
                    row[feat_name] = coef
                evolution_data.append(row)

    if not evolution_data:
        st.warning(f"No coefficient data found for {selected_action_name}.")
        return

    # Display table
    st.markdown(f"#### Coefficient Evolution for {selected_action_name}")
    evo_df = pd.DataFrame(evolution_data)

    # Format
    display_df = evo_df.copy()
    for col in display_df.columns:
        if col != 'Round':
            if model_type == 'cost' and col == 'Intercept':
                display_df[col] = display_df[col].apply(lambda x: f"{x/1e6:.4f}M")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Line chart of coefficient evolution
    st.markdown("#### Coefficient Trends Over Rounds")

    # Prepare data for plotting
    plot_data = []
    for _, row in evo_df.iterrows():
        round_id = row['Round']
        for col in evo_df.columns:
            if col != 'Round':
                plot_data.append({
                    'Round': round_id,
                    'Feature': col,
                    'Coefficient': row[col]
                })

    plot_df = pd.DataFrame(plot_data)

    fig = px.line(
        plot_df,
        x='Round',
        y='Coefficient',
        color='Feature',
        markers=True,
        title=f"Coefficient Evolution - {selected_action_name}"
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show all actions comparison at specific round
    st.markdown("---")
    st.markdown("#### All Actions Comparison")

    comparison_round = st.selectbox(
        "Compare all actions at round",
        options=sorted(all_models_data.keys()),
        index=len(all_models_data) - 1,
        key="comparison_round_select"
    )

    round_models = all_models_data[comparison_round].get('models', {})
    comparison_data = []

    for action_type, model in round_models.items():
        action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
        if hasattr(model, 'coef_'):
            coefs = model.coef_.flatten()
            intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_

            row = {'Action': action_name, 'Intercept': intercept}
            for i, coef in enumerate(coefs):
                feat_name = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i}"
                row[feat_name] = coef
            comparison_data.append(row)

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)

        # Reorder
        cols = ['Action', 'Intercept'] + [c for c in comp_df.columns if c not in ['Action', 'Intercept']]
        comp_df = comp_df[cols]

        # Format
        display_comp_df = comp_df.copy()
        for col in display_comp_df.columns:
            if col != 'Action':
                if model_type == 'cost' and col == 'Intercept':
                    display_comp_df[col] = display_comp_df[col].apply(lambda x: f"{x/1e6:.4f}M")
                else:
                    display_comp_df[col] = display_comp_df[col].apply(lambda x: f"{x:.4f}")

        st.dataframe(display_comp_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        if feature_names:
            st.markdown("#### Feature Coefficient Comparison")

            bar_data = []
            for _, row in comp_df.iterrows():
                action = row['Action']
                for feat in feature_names:
                    if feat in row:
                        bar_data.append({
                            'Action': action,
                            'Feature': feat,
                            'Coefficient': row[feat]
                        })

            bar_df = pd.DataFrame(bar_data)

            fig = px.bar(
                bar_df,
                x='Feature',
                y='Coefficient',
                color='Action',
                barmode='group',
                title=f"Feature Coefficients at Round {comparison_round}"
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
