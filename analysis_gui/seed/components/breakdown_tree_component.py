"""Lemma tree and dependency visualization using OOP data models"""
import streamlit as st
from typing import Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Breakdown, ParsedBreakdown


def render_used_lemmas_tree(breakdown: Union[Breakdown, Any]):
    """
    Render the tree of lemmas actually USED in the theorem's best proof.

    Uses the used_lemma_ids from the best proof attempt.

    Args:
        breakdown: Breakdown OOP object with parsed_breakdown
    """
    st.markdown("### 🎯 Used Lemmas in Proof")

    parsed_data = breakdown.parsed_breakdown
    if not parsed_data or not hasattr(parsed_data, 'theorem'):
        st.info("No parsed breakdown data available.")
        return

    # Get the best proof attempt from the theorem
    theorem = parsed_data.theorem
    best_attempt = theorem.get_best_attempt()

    if not best_attempt:
        st.info("No proven proof attempt found.")
        return

    # Get used lemma IDs from the best attempt
    used_lemma_ids = best_attempt.get_used_lemmas(lemmas_dict=parsed_data.lemmas, recursive=True)

    if not used_lemma_ids:
        st.info("No lemmas used in proof (direct proof).")
        return

    st.markdown(f"**Lemmas used in proof:** {len(used_lemma_ids)}")

    # Display the used lemmas with their status
    cols = st.columns(3)
    with cols[0]:
        proven = sum(1 for lid in used_lemma_ids if parsed_data.lemmas[lid].is_solved())
        st.metric("Proven", f"{proven}/{len(used_lemma_ids)}")
    with cols[1]:
        formalized = sum(1 for lid in used_lemma_ids if parsed_data.lemmas[lid].is_formalized())
        st.metric("Formalized", f"{formalized}/{len(used_lemma_ids)}")
    with cols[2]:
        st.metric("Total Used", len(used_lemma_ids))

    # Show list of used lemmas
    st.markdown("**Used Lemmas:**")
    for lemma_id in sorted(used_lemma_ids):
        lemma = parsed_data.lemmas[lemma_id]
        proven_emoji = "✅" if lemma.is_solved() else "❌"
        formalized_emoji = "✅" if lemma.is_formalized() else "❌"
        st.markdown(f"  {proven_emoji} {formalized_emoji} Lemma {lemma_id}")


def render_parsed_lemma_tree(breakdown: Union[Breakdown, Any]):
    """
    Render the parsed lemma dependency tree using the ParsedBreakdown's internal structure.

    Uses the actual dependency relationships already captured in the parsed breakdown,
    not inferred from code.

    Args:
        breakdown: Breakdown OOP object or BreakdownInfo object with compatible structure
    """
    st.markdown("### 📊 Parsed Lemma Dependency Tree")

    # Get the parsed_breakdown object (should be OOP ParsedBreakdown)
    parsed_data = breakdown.parsed_breakdown
    if not parsed_data or not hasattr(parsed_data, 'lemmas'):
        st.info("No parsed breakdown data available.")
        return

    # Get all lemmas
    lemmas_dict = parsed_data.lemmas
    if not lemmas_dict:
        st.info("No lemmas found in parsed breakdown.")
        return

    st.markdown(f"**Total lemmas identified:** {len(lemmas_dict)}")

    # Display dependency tree visualization
    try:
        # Build nodes and edges from actual dependencies field
        nodes = [{"id": "Theorem", "label": "Theorem", "level": 0}]
        edges = []

        # Add nodes for all lemmas
        for lemma_id in lemmas_dict.keys():
            nodes.append({"id": f"L{lemma_id}", "label": f"Lemma {lemma_id}", "level": 0})

        # Get theorem dependencies
        theorem = parsed_data.theorem
        theorem_deps = theorem.dependencies if hasattr(theorem, 'dependencies') else []

        # Add theorem -> lemma edges
        for lemma_id in theorem_deps:
            if lemma_id in lemmas_dict:
                edges.append({"source": "Theorem", "target": f"L{lemma_id}"})

        # Add lemma -> lemma edges from each lemma's dependencies field
        for lemma_id, lemma in lemmas_dict.items():
            lemma_dependencies = lemma.dependencies if hasattr(lemma, 'dependencies') else []
            for dep_id in lemma_dependencies:
                if dep_id in lemmas_dict:
                    edges.append({"source": f"L{lemma_id}", "target": f"L{dep_id}"})

        if not edges:
            st.info("No dependencies found in breakdown.")
            return

        st.markdown("**Dependency Tree:**")

        # Build proper tree layers (BFS from Theorem)
        layers = {}
        visited = set()
        current_layer = ["Theorem"]
        layer_num = 0

        while current_layer:
            layers[layer_num] = current_layer
            visited.update(current_layer)
            next_layer = []

            for node_id in current_layer:
                # Find direct children of this node
                children = [e["target"] for e in edges if e["source"] == node_id]
                for child in children:
                    if child not in visited and child not in next_layer:
                        next_layer.append(child)

            current_layer = next_layer
            layer_num += 1

        # Assign positions based on layers with optimal spacing
        # Use tighter vertical spacing when tree is narrow (max 2 nodes per layer)
        max_width = max(len(nodes) for nodes in layers.values())
        vert_spacing = 0.7 if max_width <= 2 else 1.2
        positions = {}
        for layer_idx, layer_nodes in layers.items():
            y = -layer_idx * vert_spacing
            num_nodes = len(layer_nodes)

            # For single nodes, place directly below their parent if they have one
            if num_nodes == 1:
                node_id = layer_nodes[0]
                # Find parent nodes
                parents = [e["source"] for e in edges if e["target"] == node_id]
                if parents and parents[0] in positions:
                    # Place directly below parent
                    x = positions[parents[0]][0]
                else:
                    x = 0
                positions[node_id] = (x, y)
            else:
                # Multiple nodes: spread evenly with more horizontal space
                max_spread = min(len(layer_nodes) * 1.2, 5)  # Dynamic spread based on number of nodes
                for node_idx, node_id in enumerate(layer_nodes):
                    if num_nodes == 1:
                        x = 0
                    else:
                        x = -(max_spread / 2) + (node_idx * max_spread / (num_nodes - 1))
                    positions[node_id] = (x, y)

        # Create figure with more space for depth and complex trees
        num_layers = len(layers)
        max_nodes_in_layer = max(len(nodes) for nodes in layers.values())
        fig_height = max(5, num_layers * 1.5)  # Scale height based on number of layers
        fig_width = max(8, max_nodes_in_layer * 2)  # Scale width based on widest layer
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Group edges by source to space out arrows from same parent
        edges_by_source = {}
        for edge in edges:
            source = edge["source"]
            if source not in edges_by_source:
                edges_by_source[source] = []
            edges_by_source[source].append(edge)

        # Draw edges with arrows (orthogonal routing to avoid overlaps)
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]

                # Get index of this edge among edges from same source
                edges_from_source = edges_by_source[source]
                source_edge_idx = edges_from_source.index(edge)
                num_from_source = len(edges_from_source)

                # Space out the starting points of arrows from same source
                if num_from_source == 1:
                    start_offset = 0
                else:
                    start_offset = (source_edge_idx - (num_from_source - 1) / 2) * 0.15

                # Draw straight arrow from source to target
                # Calculate direction and offset by radius to start/end at circle edge
                dx = x2 - x1
                dy = y2 - y1
                dist = (dx**2 + dy**2)**0.5
                if dist > 0:
                    # Normalize direction
                    norm_dx = dx / dist
                    norm_dy = dy / dist
                    # Start at edge of source circle (radius 0.22 for theorem, 0.20 for lemmas)
                    src_radius = 0.22 if source == "Theorem" else 0.20
                    tgt_radius = 0.22 if target == "Theorem" else 0.20
                    # Arrow starts at source circle edge and ends at target circle edge
                    start_x = x1 + norm_dx * src_radius
                    start_y = y1 + norm_dy * src_radius
                    end_x = x2 - norm_dx * tgt_radius
                    end_y = y2 - norm_dy * tgt_radius
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', lw=1.2, color='black', alpha=0.5))

        # Draw nodes as circles
        for node in nodes:
            if node["id"] not in positions:
                continue

            x, y = positions[node["id"]]

            # Different colors for root and lemmas
            if node["id"] == "Theorem":
                color = '#FF6B6B'
                radius = 0.22
                label = 'T'
            else:
                color = '#4ECDC4'
                radius = 0.20
                # Extract lemma number from node id (e.g., "L1" -> 1)
                label = node["id"]  # Already in format "L1", "L2", etc.

            # Draw circle
            circle = mpatches.Circle((x, y), radius, color=color, ec='black', lw=0.8, zorder=3)
            ax.add_patch(circle)

            # Add label
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=4)

        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')

        # Set tight limits
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
            ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        st.pyplot(fig, width="content")
        plt.close(fig)
    except Exception as e:
        st.warning(f"⚠️ Could not render dependency tree: {str(e)}")
