"""
Embeddings analysis component - visualize and analyze Qwen3 embeddings of breakdowns.

Displays cosine similarity matrices for reasoning traces and outputs,
organized by problem and breakdown, with text preview capabilities.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import re


@st.cache_data
def load_embeddings(run_dir: str, embedding_type: str) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Load embeddings and metadata from JSON files.

    Args:
        run_dir: Path to the run directory
        embedding_type: Either "output" or "reasoning_trace"

    Returns:
        Tuple of (data_records, embedding_vectors, metadata)
    """
    embeddings_dir = Path(run_dir) / "embeddings"

    # Load data
    data_file = embeddings_dir / f"{embedding_type}_data.json"
    embeddings_file = embeddings_dir / f"{embedding_type}_embeddings.json"
    metadata_file = embeddings_dir / f"{embedding_type}_metadata.json"

    with open(data_file) as f:
        data = json.load(f)

    with open(embeddings_file) as f:
        embeddings_data = json.load(f)

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Convert embeddings to numpy array
    if isinstance(embeddings_data, dict) and "embeddings" in embeddings_data:
        embedding_vectors = np.array(embeddings_data["embeddings"])
    elif isinstance(embeddings_data, list):
        embedding_vectors = np.array(embeddings_data)
    else:
        raise ValueError(f"Unexpected embeddings format: {type(embeddings_data)}")

    return data, embedding_vectors, metadata


def parse_uid(uid: str) -> Dict[str, any]:
    """
    Parse UID to extract metadata.
    Format: "problem_id_r0_b0" or similar
    """
    # Match pattern: origin_problem_id_r{round}_b{breakdown}
    match = re.match(r"^(.+?)_r(\d+)_b(\d+)$", uid)
    if match:
        return {
            "origin_problem_id": match.group(1),
            "round_id": int(match.group(2)),
            "breakdown_id": int(match.group(3))
        }
    return {"origin_problem_id": uid, "round_id": 0, "breakdown_id": 0}


def sort_embeddings(data: Dict, embedding_vectors: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    """
    Sort embeddings and data by origin_problem_id and breakdown_id.

    Returns:
        Tuple of (sorted_records, sorted_embeddings)
    """
    records = data.get("records", [])

    # Parse UIDs and add parsed metadata
    for record in records:
        parsed = parse_uid(record["uid"])
        record["parsed_metadata"] = parsed

    # Sort by origin_problem_id, then breakdown_id
    sorted_records = sorted(
        records,
        key=lambda r: (r["parsed_metadata"]["origin_problem_id"], r["parsed_metadata"]["breakdown_id"])
    )

    # Reorder embeddings to match sorted records
    sorted_indices = [records.index(r) for r in sorted_records]
    sorted_embeddings = embedding_vectors[sorted_indices]

    return sorted_records, sorted_embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for embeddings."""
    return cosine_similarity(embeddings)


def create_heatmap(similarity_matrix: np.ndarray, labels: List[str], title: str, show_text: bool = True) -> go.Figure:
    """
    Create interactive heatmap for similarity matrix.

    Args:
        similarity_matrix: Cosine similarity matrix
        labels: Labels for rows/columns (UIDs)
        title: Title for the heatmap
        show_text: Whether to show similarity values as text (default False for full view)

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmid=0.5,
        text=np.round(similarity_matrix, 3) if show_text else None,
        texttemplate='%{text}' if show_text else None,
        textfont={"size": 8} if show_text else None,
        hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Breakdown",
        yaxis_title="Breakdown",
        height=max(600, len(labels) * 15),
        width=max(600, len(labels) * 15),
        hovermode='closest'
    )

    # Make axes readable
    fig.update_xaxes(tickangle=45)

    return fig


def render_embeddings_analysis(run_dir: str):
    """
    Render embeddings analysis interface.

    Args:
        run_dir: Path to the run directory
    """
    st.header("🔍 Embeddings Analysis")
    st.markdown("Analyze Qwen3 embeddings of problem breakdowns and reasoning traces")

    try:
        # Load embeddings for both types
        output_data, output_embeddings, output_metadata = load_embeddings(run_dir, "output")
        trace_data, trace_embeddings, trace_metadata = load_embeddings(run_dir, "reasoning_trace")

        # Sort embeddings
        sorted_output_records, sorted_output_emb = sort_embeddings(output_data, output_embeddings)
        sorted_trace_records, sorted_trace_emb = sort_embeddings(trace_data, trace_embeddings)

        # Create similarity matrices
        output_sim = compute_similarity_matrix(sorted_output_emb)
        trace_sim = compute_similarity_matrix(sorted_trace_emb)

        # Tab selection
        tab1, tab2, tab3 = st.tabs(["📊 Output Similarity", "📊 Reasoning Trace Similarity", "🔎 Per-Problem Analysis"])

        with tab1:
            render_similarity_matrix_view(sorted_output_records, output_sim, "Breakdown Output")

        with tab2:
            render_similarity_matrix_view(sorted_trace_records, trace_sim, "Reasoning Trace")

        with tab3:
            render_per_problem_view(sorted_output_records, sorted_output_emb, sorted_trace_records, sorted_trace_emb)

    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def render_similarity_matrix_view(records: List[Dict], similarity_matrix: np.ndarray, embedding_type: str):
    """Render full similarity matrix view."""
    st.subheader(f"{embedding_type} Similarity Matrix")
    st.markdown(f"Cosine similarity between all {embedding_type.lower()} embeddings (sorted by problem ID and breakdown ID)")

    # Create labels
    labels = [f"{r['parsed_metadata']['origin_problem_id']}<br>b{r['parsed_metadata']['breakdown_id']}" for r in records]

    # Create heatmap without text
    fig = create_heatmap(similarity_matrix, labels, f"{embedding_type} Cosine Similarity Matrix", show_text=False)
    st.plotly_chart(fig, use_container_width=True)

    # Collect similarities between breakdowns of the same problem only
    within_problem_similarities = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            problem_i = records[i]["parsed_metadata"]["origin_problem_id"]
            problem_j = records[j]["parsed_metadata"]["origin_problem_id"]
            if problem_i == problem_j:
                within_problem_similarities.append(similarity_matrix[i, j])

    within_problem_similarities = np.array(within_problem_similarities)

    # Display histogram right below heatmap (only within-problem similarities)
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=within_problem_similarities,
        nbinsx=20,
        name=f"{embedding_type} Similarity",
        marker_color="rgba(0, 100, 200, 0.7)"
    ))
    if len(within_problem_similarities) > 0:
        hist_fig.add_vline(x=within_problem_similarities.mean(), line_dash="dash", line_color="red",
                           annotation_text=f"Mean: {within_problem_similarities.mean():.3f}")
    hist_fig.update_layout(
        title=f"{embedding_type} Similarity Distribution (Within-Problem Pairs Only)",
        xaxis_title="Cosine Similarity",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Show statistics
    st.subheader("Similarity Statistics (Within-Problem Pairs)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Similarity", f"{within_problem_similarities.mean():.3f}" if len(within_problem_similarities) > 0 else "N/A")
    with col2:
        st.metric("Max Similarity", f"{within_problem_similarities.max():.3f}" if len(within_problem_similarities) > 0 else "N/A")
    with col3:
        st.metric("Min Similarity", f"{within_problem_similarities.min():.3f}" if len(within_problem_similarities) > 0 else "N/A")
    with col4:
        st.metric("Std Dev", f"{within_problem_similarities.std():.3f}" if len(within_problem_similarities) > 0 else "N/A")

    # Show high similarity pairs
    st.subheader("Highest Similarity Pairs")
    similarity_pairs = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            similarity_pairs.append({
                "Breakdown 1": records[i]["uid"],
                "Breakdown 2": records[j]["uid"],
                "Similarity": similarity_matrix[i, j]
            })

    similarity_df = pd.DataFrame(similarity_pairs).sort_values("Similarity", ascending=False)
    st.dataframe(similarity_df.head(20), use_container_width=True)


def get_off_diagonal_similarities(similarity_matrix: np.ndarray) -> np.ndarray:
    """Extract off-diagonal elements (exclude diagonal) from similarity matrix."""
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    return similarity_matrix[mask]


def render_per_problem_view(output_records: List[Dict], output_emb: np.ndarray,
                           trace_records: List[Dict], trace_emb: np.ndarray):
    """Render per-problem detailed analysis view."""
    st.subheader("Per-Problem Analysis")
    st.markdown("Select a problem to see similarity between its breakdowns and view breakdown details")

    # Get unique problems
    problems = sorted(set(r["parsed_metadata"]["origin_problem_id"] for r in output_records))

    selected_problem = st.selectbox("Select Problem", problems, key="problem_select")

    if not selected_problem:
        st.info("Select a problem to view details")
        return

    # Filter records for this problem
    output_problem_records = [r for r in output_records
                             if r["parsed_metadata"]["origin_problem_id"] == selected_problem]
    trace_problem_records = [r for r in trace_records
                            if r["parsed_metadata"]["origin_problem_id"] == selected_problem]

    # Get indices in original arrays
    output_indices = [output_records.index(r) for r in output_problem_records]
    trace_indices = [trace_records.index(r) for r in trace_problem_records]

    # Extract submatrices
    output_problem_emb = output_emb[np.ix_(output_indices, output_indices)]
    trace_problem_emb = trace_emb[np.ix_(trace_indices, trace_indices)]

    # Compute similarities
    output_problem_sim = cosine_similarity(output_problem_emb)
    trace_problem_sim = cosine_similarity(trace_problem_emb)

    # Get off-diagonal similarities (between different breakdowns)
    output_off_diag = get_off_diagonal_similarities(output_problem_sim)
    trace_off_diag = get_off_diagonal_similarities(trace_problem_sim)

    # Display statistics and histograms
    st.subheader("Similarity Statistics Within Problem")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown("### Output Embeddings")
        st.metric("Average Similarity", f"{output_off_diag.mean():.3f}")
        st.metric("Min Similarity", f"{output_off_diag.min():.3f}")
        st.metric("Max Similarity", f"{output_off_diag.max():.3f}")
        st.metric("Std Dev", f"{output_off_diag.std():.3f}")

    with stat_col2:
        st.markdown("### Reasoning Trace Embeddings")
        st.metric("Average Similarity", f"{trace_off_diag.mean():.3f}")
        st.metric("Min Similarity", f"{trace_off_diag.min():.3f}")
        st.metric("Max Similarity", f"{trace_off_diag.max():.3f}")
        st.metric("Std Dev", f"{trace_off_diag.std():.3f}")

    # Display histograms
    st.markdown("---")
    st.subheader("Similarity Distribution Within Problem")

    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        # Output histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=output_off_diag,
            nbinsx=15,
            name="Output Similarity",
            marker_color="rgba(0, 100, 200, 0.7)"
        ))
        fig.add_vline(x=output_off_diag.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {output_off_diag.mean():.3f}")
        fig.update_layout(
            title=f"{selected_problem}: Output Similarity Distribution",
            xaxis_title="Cosine Similarity",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with hist_col2:
        # Reasoning trace histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trace_off_diag,
            nbinsx=15,
            name="Trace Similarity",
            marker_color="rgba(0, 150, 100, 0.7)"
        ))
        fig.add_vline(x=trace_off_diag.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {trace_off_diag.mean():.3f}")
        fig.update_layout(
            title=f"{selected_problem}: Reasoning Trace Similarity Distribution",
            xaxis_title="Cosine Similarity",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Similarity Heatmaps")

    # Display heatmaps
    heatmap_col1, heatmap_col2 = st.columns(2)

    with heatmap_col1:
        st.markdown("#### Output Similarity")
        labels = [f"b{r['parsed_metadata']['breakdown_id']}" for r in output_problem_records]
        fig = create_heatmap(output_problem_sim, labels, f"Output Similarity", show_text=False)
        st.plotly_chart(fig, use_container_width=True)

    with heatmap_col2:
        st.markdown("#### Reasoning Trace Similarity")
        labels = [f"b{r['parsed_metadata']['breakdown_id']}" for r in trace_problem_records]
        fig = create_heatmap(trace_problem_sim, labels, f"Reasoning Trace Similarity", show_text=False)
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown comparison
    st.markdown("---")
    st.subheader("Breakdown Details & Comparison")

    # Select which breakdowns to compare
    breakdown_ids = sorted(set(r["parsed_metadata"]["breakdown_id"] for r in output_problem_records))
    selected_breakdowns = st.multiselect(
        "Select breakdowns to view",
        breakdown_ids,
        default=breakdown_ids[:2] if len(breakdown_ids) >= 2 else breakdown_ids,
        key=f"breakdowns_{selected_problem}"
    )

    if selected_breakdowns:
        # Display side-by-side comparison
        cols = st.columns(len(selected_breakdowns))

        for idx, breakdown_id in enumerate(selected_breakdowns):
            output_rec = next((r for r in output_problem_records
                             if r["parsed_metadata"]["breakdown_id"] == breakdown_id), None)

            with cols[idx]:
                st.markdown(f"### Breakdown {breakdown_id}")
                st.markdown(f"**UID:** `{output_rec['uid']}`")

                if output_rec:
                    # Show full text in markdown
                    text = output_rec.get("text", "")
                    st.markdown(text)

        # Show similarity between selected breakdowns
        st.markdown("---")
        st.subheader("Similarity Between Selected Breakdowns")

        if len(selected_breakdowns) > 1:
            # Create comparison table
            comparison_data = []
            for i, bid1 in enumerate(selected_breakdowns):
                for bid2 in selected_breakdowns[i+1:]:
                    idx1 = breakdown_ids.index(bid1)
                    idx2 = breakdown_ids.index(bid2)
                    output_sim = output_problem_sim[idx1, idx2]
                    trace_sim = trace_problem_sim[idx1, idx2]
                    comparison_data.append({
                        "Breakdown 1": f"b{bid1}",
                        "Breakdown 2": f"b{bid2}",
                        "Output Similarity": f"{output_sim:.4f}",
                        "Trace Similarity": f"{trace_sim:.4f}"
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Select at least 2 breakdowns to compare")
