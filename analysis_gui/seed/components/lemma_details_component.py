"""Lemma list and details rendering
"""
import streamlit as st
from typing import List, Dict, Any, Optional
from utils import truncate_text, extract_lemma_dependencies, build_dependency_tree_data

# Import utilities from other components
from .proof_status_component import extract_think_content, count_axioms_in_code, render_proof_attempt_expandables
from .formalization_component import render_compilation_result
from .theorem_viewer_component import render_lemma_attempts_from_parsed_breakdown
from seed_data_models import Breakdown

def render_lemma_list(breakdown: Breakdown, analysis: Optional[Dict[str, Any]] = None):
    """
    Render only the lemma list with details.

    Args:
        breakdown: Breakdown object
        analysis: Optional ProblemAnalysis object
    """
    st.markdown("### 📋 Lemma List")

    # Get the parsed_breakdown object (should be OOP ParsedBreakdown)
    parsed_data = breakdown.parsed_breakdown
    if not parsed_data or not hasattr(parsed_data, 'lemmas'):
        st.info("No parsed breakdown data available.")
        return

    # Extract lemmas from OOP ParsedBreakdown object
    lemmas = list(parsed_data.lemmas.values())

    if not lemmas:
        st.info("No lemmas found in parsed breakdown.")
        return

    # Get lemma prover stats if available
    lemma_stats = None
    if analysis:
        lemma_results = [
            bd for bd in analysis.lemma_prover_stats.get('breakdown_results', [])
            if bd.get('breakdown_id') == breakdown.problem_id
        ]
        lemma_stats = lemma_results[0] if lemma_results else None

    # Get parsed breakdown if available (should be OOP ParsedBreakdown)
    parsed_data = breakdown.parsed_breakdown

    # Build a map of proven lemmas from lemma prover results
    proven_lemma_ids = set()
    attempted_lemma_ids = set()

    # First, try to get from OOP parsed_breakdown (minified data)
    if parsed_data and hasattr(parsed_data, 'lemmas'):
        for lemma_id, lemma in parsed_data.lemmas.items():
            # A lemma is attempted if it has formalizations with proof attempts
            if hasattr(lemma, 'formalizations') and lemma.formalizations:
                for formalization in lemma.formalizations:
                    if hasattr(formalization, 'proof_attempts') and formalization.proof_attempts:
                        attempted_lemma_ids.add(lemma_id)
                        break

            # Check if lemma is solved (has a passing & complete proof or recursive attempt)
            if hasattr(lemma, 'is_solved') and lemma.is_solved():
                proven_lemma_ids.add(lemma_id)

    # Fall back to legacy lemma_prover_results if no OOP data
    if not parsed_data or not hasattr(parsed_data, 'lemmas'):
        if breakdown.lemma_prover_results and breakdown.lemma_prover_results.get('lemmas'):
            for lemma_id, lemma_results in breakdown.lemma_prover_results.get('lemmas', {}).items():
                if lemma_results.get('attempts'):
                    attempted_lemma_ids.add(lemma_id)
                    # Check if any attempt passed AND is complete (no sorries)
                    for attempt in lemma_results.get('attempts', []):
                        data = attempt.get('data', {})
                        comp_result = data.get('compilation_result', {})
                        if comp_result.get('pass', False) and comp_result.get('complete', False):
                            proven_lemma_ids.add(lemma_id)
                            break  # Only need one successful attempt

    # Display lemmas in expanders
    for idx, lemma in enumerate(lemmas, 1):
        # OOP Lemma object - direct attribute access
        lemma_statement = lemma.statement or f'Lemma {idx}'
        lemma_id_full = lemma.lemma_id
        lemma_assumptions = lemma.assumptions

        # Show lemma with status - check if this specific lemma was attempted and/or proven
        lemma_display = f"**Lemma {lemma_id_full}:** {truncate_text(lemma_statement, 100)}"
        lemma_is_proven = lemma_id_full in proven_lemma_ids
        lemma_attempted = lemma_id_full in attempted_lemma_ids

        # Use different emoji based on status
        if lemma_is_proven:
            status_emoji = "✅"  # Proven and complete
        elif lemma_attempted:
            status_emoji = "❌"  # Attempted but failed
        else:
            status_emoji = "⏭️"  # Not attempted (recursive pruning)

        lemma_display += f" {status_emoji}"

        with st.expander(lemma_display, expanded=False):
            # Extract and show lemma dependencies
            combined_text = ""
            if lemma_statement:
                combined_text += lemma_statement + "\n"
            if lemma_assumptions:
                combined_text += lemma_assumptions + "\n"

            dependencies = extract_lemma_dependencies(combined_text)
            if dependencies:
                dep_labels = [f"Lemma {d}" for d in dependencies]
                st.markdown(f"**Depends on:** {', '.join(dep_labels)}")
                st.markdown("---")

            # Show full statement
            st.markdown("**Statement:**")
            st.markdown(lemma_statement)

            # Show assumption if available
            if lemma_assumptions:
                st.markdown("**Assumption:**")
                st.markdown(lemma_assumptions)

            # Show formalization if available
            # Pass the full lemma ID for prover results filtering
            render_lemma_formalization(lemma, lemma_id_full, parsed_data, breakdown)

    # Show overall lemma stats
    if lemma_stats:
        st.markdown("---")
        st.markdown("### Lemma Prover Results")



def render_lemmas_section(
    breakdown: Breakdown,
    analysis: Optional[Dict[str, Any]] = None
):
    """
    Display lemmas from parsed breakdown data.

    Args:
        breakdown: Breakdown object
        analysis: ProblemAnalysis object (optional)
    """
    st.markdown("### 📚 Lemmas from Parsed Breakdown")

    # Get the parsed_breakdown object (should be OOP ParsedBreakdown)
    parsed_data = breakdown.parsed_breakdown
    if not parsed_data or not hasattr(parsed_data, 'lemmas'):
        st.info("No parsed breakdown data available.")
        return

    # Extract lemmas from OOP ParsedBreakdown object
    lemmas = list(parsed_data.lemmas.values())

    if not lemmas:
        st.info("No lemmas found in parsed breakdown.")
        return

    st.markdown(f"**Total lemmas identified:** {len(lemmas)}")

    # Display dependency tree visualization
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Extract theorem text to detect its dependencies from OOP structure
        parsed_data_for_theorem = breakdown.parsed_breakdown

        theorem_text = ""
        if parsed_data_for_theorem and hasattr(parsed_data_for_theorem, 'theorem'):
            # OOP ParsedBreakdown object
            theorem_obj = parsed_data_for_theorem.theorem
            if theorem_obj:
                # Combine statement from theorem
                theorem_text = (theorem_obj.statement or '') + "\n"
                if hasattr(theorem_obj, 'proof_idea') and theorem_obj.proof_idea:
                    theorem_text += theorem_obj.proof_idea

        root, nodes, edges = build_dependency_tree_data(lemmas, theorem_text)
        if root and nodes:
            st.markdown("**Dependency Tree:**")

            # Debug output: show what lemmas the theorem depends on and inter-lemma dependencies
            with st.expander("📋 Debug: Tree Structure", expanded=False):
                st.write("**Theorem dependencies:**")
                theorem_deps = [e["target"].replace("L", "") for e in edges if e["source"] == "Theorem"]
                if theorem_deps:
                    st.write(f"Theorem → Lemmas: {', '.join(theorem_deps)}")
                else:
                    st.write("Theorem → (no direct lemma dependencies)")

                st.write("**Lemma-to-lemma dependencies:**")
                lemma_to_lemma = [f"Lemma {e['source'].replace('L', '')} → Lemma {e['target'].replace('L', '')}"
                                 for e in edges if e["source"] != "Theorem"]
                if lemma_to_lemma:
                    for dep in lemma_to_lemma:
                        st.write(dep)
                else:
                    st.write("(no lemma-to-lemma dependencies)")

                st.write(f"**Total nodes:** {len(nodes)}")
                st.write(f"**Total edges:** {len(edges)}")

            # Filter redundant edges: remove direct theorem→lemma edges if that lemma
            # is also reachable through other edges (to reduce clutter)
            filtered_edges = []

            # First, find all lemmas that the theorem depends on directly
            theorem_direct_lemmas = {e["target"] for e in edges if e["source"] == "Theorem"}

            # Then find all lemmas that are reachable through other paths
            lemma_to_lemma_edges = [e for e in edges if e["source"] != "Theorem"]

            # Build a graph of lemma dependencies to find reachable nodes
            lemma_graph = {}
            for e in lemma_to_lemma_edges:
                if e["source"] not in lemma_graph:
                    lemma_graph[e["source"]] = []
                lemma_graph[e["source"]].append(e["target"])

            # Find lemmas reachable from any directly-used lemma
            reachable_from_direct = set()
            for direct_lemma in theorem_direct_lemmas:
                # BFS to find all reachable lemmas from this direct dependency
                queue = [direct_lemma]
                visited_bfs = {direct_lemma}
                while queue:
                    current = queue.pop(0)
                    if current in lemma_graph:
                        for neighbor in lemma_graph[current]:
                            if neighbor not in visited_bfs:
                                visited_bfs.add(neighbor)
                                queue.append(neighbor)
                                reachable_from_direct.add(neighbor)

            # Keep theorem edges and lemma-to-lemma edges
            filtered_edges = lemma_to_lemma_edges.copy()

            # Only add theorem edges for lemmas not already reachable
            for e in edges:
                if e["source"] == "Theorem":
                    # Only add if this lemma is not reachable from other theorem dependencies
                    if e["target"] not in reachable_from_direct:
                        filtered_edges.append(e)

            edges = filtered_edges

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
            positions = {}
            for layer_idx, layer_nodes in layers.items():
                y = -layer_idx * 1.2  # Increased vertical spacing between layers for more depth
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
            for edge_idx, edge in enumerate(edges):
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
                if node["level"] == 0:
                    color = '#FF6B6B'
                    radius = 0.22
                    label = 'T'
                else:
                    color = '#4ECDC4'
                    radius = 0.20
                    # Extract lemma number from node id (e.g., "L1" -> "L1")
                    lemma_num = node["id"].replace('L', '')
                    label = f'L{lemma_num}'

                # Draw circle
                circle = mpatches.Circle((x, y), radius, color=color, ec='black', lw=0.8, zorder=3)
                ax.add_patch(circle)

                # Add label
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white', zorder=4)

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

    st.markdown("---")

    # Get lemma prover stats if available
    lemma_stats = None
    if analysis:
        lemma_results = [
            bd for bd in analysis.lemma_prover_stats.get('breakdown_results', [])
            if bd.get('breakdown_id') == breakdown.problem_id
        ]
        lemma_stats = lemma_results[0] if lemma_results else None

    # Get parsed breakdown if available (should be OOP ParsedBreakdown)
    parsed_data = breakdown.parsed_breakdown

    # Build a map of proven lemmas from lemma prover results
    proven_lemma_ids = set()
    attempted_lemma_ids = set()

    # First, try to get from OOP parsed_breakdown (minified data)
    if parsed_data and hasattr(parsed_data, 'lemmas'):
        for lemma_id, lemma in parsed_data.lemmas.items():
            # A lemma is attempted if it has formalizations with proof attempts
            if hasattr(lemma, 'formalizations') and lemma.formalizations:
                for formalization in lemma.formalizations:
                    if hasattr(formalization, 'proof_attempts') and formalization.proof_attempts:
                        attempted_lemma_ids.add(lemma_id)
                        break

            # Check if lemma is solved (has a passing & complete proof or recursive attempt)
            if hasattr(lemma, 'is_solved') and lemma.is_solved():
                proven_lemma_ids.add(lemma_id)

    # Fall back to legacy lemma_prover_results if no OOP data
    if not parsed_data or not hasattr(parsed_data, 'lemmas'):
        if breakdown.lemma_prover_results and breakdown.lemma_prover_results.get('lemmas'):
            for lemma_id, lemma_results in breakdown.lemma_prover_results.get('lemmas', {}).items():
                if lemma_results.get('attempts'):
                    attempted_lemma_ids.add(lemma_id)
                    # Check if any attempt passed AND is complete (no sorries)
                    for attempt in lemma_results.get('attempts', []):
                        data = attempt.get('data', {})
                        comp_result = data.get('compilation_result', {})
                        if comp_result.get('pass', False) and comp_result.get('complete', False):
                            proven_lemma_ids.add(lemma_id)
                            break  # Only need one successful attempt

    # Display lemmas in expanders
    for idx, lemma in enumerate(lemmas, 1):
        # OOP Lemma object - direct attribute access
        lemma_statement = lemma.statement or f'Lemma {idx}'
        lemma_id_full = lemma.lemma_id
        lemma_assumptions = lemma.assumptions

        # Show lemma with status - check if this specific lemma was attempted and/or proven
        lemma_display = f"**Lemma {lemma_id_full}:** {truncate_text(lemma_statement, 100)}"
        lemma_is_proven = lemma_id_full in proven_lemma_ids
        lemma_attempted = lemma_id_full in attempted_lemma_ids

        # Use different emoji based on status
        if lemma_is_proven:
            status_emoji = "✅"  # Proven and complete
        elif lemma_attempted:
            status_emoji = "❌"  # Attempted but failed
        else:
            status_emoji = "⏭️"  # Not attempted (recursive pruning)

        lemma_display += f" {status_emoji}"

        with st.expander(lemma_display, expanded=False):
            # Extract and show lemma dependencies
            combined_text = ""
            if lemma_statement:
                combined_text += lemma_statement + "\n"
            if lemma_assumptions:
                combined_text += lemma_assumptions + "\n"

            dependencies = extract_lemma_dependencies(combined_text)
            if dependencies:
                dep_labels = [f"Lemma {d}" for d in dependencies]
                st.markdown(f"**Depends on:** {', '.join(dep_labels)}")
                st.markdown("---")

            # Show full statement
            st.markdown("**Statement:**")
            st.markdown(lemma_statement)

            # Show assumption if available
            if lemma_assumptions:
                st.markdown("**Assumption:**")
                st.markdown(lemma_assumptions)

            # Show formalization if available
            # Pass the full lemma ID for prover results filtering
            render_lemma_formalization(lemma, lemma_id_full, parsed_data, breakdown)

    # Show overall lemma stats
    if lemma_stats:
        st.markdown("---")
        st.markdown("### Lemma Prover Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lemmas", lemma_stats.get('total_lemmas', 0))
        with col2:
            st.metric("Proven", lemma_stats.get('proven_lemmas', 0))
        with col3:
            status = "✅ All Proven" if lemma_stats.get('all_lemmas_proven', False) else "❌ Not All Proven"
            st.markdown(f"**Status:** {status}")



def render_formalization_detail(formalization, formalization_id: int):
    """
    Render a single formalization with its details.

    Args:
        formalization: A Formalization object
        formalization_id: The ID of this formalization (for display)
    """
    # Show formalization reasoning if available
    formalization_reasoning = formalization.formalization_reasoning
    if formalization_reasoning:
        with st.expander("💭 Formalization Reasoning"):
            st.markdown(formalization_reasoning)

    # Show formal statement if available
    formal_statement = formalization.formal_statement
    if formal_statement:
        st.markdown("**Formal Statement (Lean 4):**")
        cleaned_code = formal_statement.strip()
        st.code(cleaned_code, language="lean")

    # Show compilation and validation results
    st.markdown("**Formalization Status:**")
    col1, col2 = st.columns(2)

    with col1:
        status = "✅ Pass" if formalization.compilation_pass else "❌ Failed"
        st.metric("Compilation", status)

    with col2:
        status = "✅ Pass" if formalization.validation_pass else "❌ Failed"
        st.metric("Validation", status)

    # Show detailed compilation results if available
    compilation_result = formalization.compilation_result
    if compilation_result:
        with st.expander("📋 Compilation Output", expanded=False):
            if isinstance(compilation_result, dict):
                st.json(compilation_result)
            else:
                st.code(str(compilation_result), language="text")

    # Show detailed validation results if available or if validation failed
    validation_result = formalization.validation_result
    expanded = not formalization.validation_pass  # Expand if validation failed
    if validation_result or (not formalization.validation_pass):
        with st.expander("✓ Validation Output", expanded=expanded):
            if validation_result:
                if isinstance(validation_result, dict):
                    st.json(validation_result)
                else:
                    st.code(str(validation_result), language="text")
            else:
                st.info("No validation output available")

    # Show compilation errors if available
    compilation_errors = formalization.compilation_errors
    if compilation_errors:
        st.error("**Compilation Errors:**")
        if isinstance(compilation_errors, list):
            for error in compilation_errors:
                st.code(str(error), language="text")
        else:
            st.code(str(compilation_errors), language="text")


def render_lemma_formalization(lemma, lemma_id_full, parsed_data: Optional[Dict[str, Any]], breakdown: Optional[Breakdown] = None):
    """
    Render formalization details for a lemma including Lean code, compilation results, and prover attempts.
    Supports multiple formalizations with tabs to switch between them.

    Args:
        lemma: The lemma data from parsed breakdown (Lemma OOP object)
        lemma_id_full: The lemma ID as integer (from parsed breakdown)
        parsed_data: The parsed breakdown data (ParsedBreakdown OOP object or dict)
        breakdown: The Breakdown object for accessing lemma prover results
    """
    if not parsed_data:
        return

    # Handle both OOP ParsedBreakdown and legacy dict structures
    parsed_lemma = None

    if hasattr(parsed_data, 'lemmas'):
        # OOP ParsedBreakdown - match by integer lemma_id
        if isinstance(lemma_id_full, int):
            parsed_lemma = parsed_data.lemmas.get(lemma_id_full)
        else:
            # If lemma_id_full is a full ID string, try to extract the integer
            import re
            match = re.search(r'_l(\d+)', str(lemma_id_full))
            if match:
                lemma_num = int(match.group(1))
                parsed_lemma = parsed_data.lemmas.get(lemma_num)

    if not parsed_lemma:
        return

    st.markdown("---")
    st.markdown("### ⚙️ Formalization")

    # Check if we have multiple formalizations (new structure)
    if hasattr(parsed_lemma, 'formalizations') and len(parsed_lemma.formalizations) > 1:
        # New structure with multiple formalizations - use tabs
        # Create tabs for each formalization with labels like "0", "1", "2", "3"
        # Add ✅ emoji if formalization is selected (in selected_formalizations)
        tab_labels = [
            f"{'✅ ' if form.is_selected else ''}{form.id}" if form.id is not None else f"{'✅ ' if form.is_selected else ''}{i}"
            for i, form in enumerate(parsed_lemma.formalizations)
        ]
        tabs = st.tabs(tab_labels)

        for tab, form, idx in zip(tabs, parsed_lemma.formalizations, range(len(parsed_lemma.formalizations))):
            with tab:
                # Add status badge at top of tab
                status_cols = st.columns(3)
                with status_cols[0]:
                    comp_emoji = "✅" if form.compilation_pass else "❌"
                    st.markdown(f"**Compilation:** {comp_emoji}")
                with status_cols[1]:
                    val_emoji = "✅" if form.validation_pass else "❌"
                    st.markdown(f"**Validation:** {val_emoji}")
                with status_cols[2]:
                    proven_emoji = "✅" if form.is_proven() else "⏳"
                    st.markdown(f"**Proven:** {proven_emoji}")

                st.markdown("---")

                # Render the detailed formalization
                render_formalization_detail(form, idx)

    elif hasattr(parsed_lemma, 'formalizations') and len(parsed_lemma.formalizations) == 1:
        # Single formalization - show directly
        best_formalization = parsed_lemma.formalizations[0]
        render_formalization_detail(best_formalization, 0)

    elif hasattr(parsed_lemma, 'get_best_formalization'):
        # Fallback: old-style single best formalization
        best_formalization = parsed_lemma.get_best_formalization()
        if best_formalization:
            render_formalization_detail(best_formalization, 0)

    # Show lemma proof attempts from parsed_breakdown (minified data) or legacy lemma_prover_results
    if breakdown:
        # Try new data model first (minified)
        if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
            st.markdown("---")

            # Load attempts button for lemma
            col1, col2, col3 = st.columns([2, 2, 6])
            with col1:
                if st.button("📥 Load Attempts", key=f"load_attempts_lemma_{breakdown.origin_problem_id}_{breakdown.round_id}_{breakdown.breakdown_id}_{lemma_id_full}", help="Load full proof attempts for this lemma"):
                    _handle_load_attempts_lemma(breakdown, lemma_id=lemma_id_full)

            render_lemma_attempts_from_parsed_breakdown(breakdown, lemma_id_full)
        # Fall back to legacy structure
        elif breakdown.lemma_prover_results:
            lemmas_dict = breakdown.lemma_prover_results.get('lemmas', {})
            lemma_results = lemmas_dict.get(lemma_id_full)

            if lemma_results:
                attempts = lemma_results.get('attempts', [])
                if attempts:
                    st.markdown("---")
                    st.markdown("### 🤖 Lemma Prover Results")

                    # Sort attempts by correction round
                    attempts_sorted = sorted(
                        attempts,
                        key=lambda x: (x.get('correction_round', 0), x.get('data', {}).get('name', ''))
                    )

                    # Check if any attempt passed
                    any_passed = any(
                        attempt.get('data', {}).get('compilation_result', {}).get('pass', False) and
                        attempt.get('data', {}).get('compilation_result', {}).get('complete', False)
                        for attempt in attempts_sorted
                    )

                    # Display overall status
                    col1, col2 = st.columns(2)
                    with col1:
                        status_emoji = "✅" if any_passed else "❌"
                        status_text = "Proven" if any_passed else "Not Proven"
                        st.metric("Status", f"{status_emoji} {status_text}")
                    with col2:
                        st.metric("Total Attempts", len(attempts_sorted))

                    # Display attempts in dropdown
                    with st.expander("📋 View All Proof Attempts"):
                        for round_num in sorted(set(a.get('correction_round', 0) for a in attempts_sorted)):
                            round_attempts = [a for a in attempts_sorted if a.get('correction_round') == round_num]

                            # Count passed/failed
                            passed_count = sum(
                                1 for a in round_attempts
                                if a.get('data', {}).get('compilation_result', {}).get('pass', False)
                            )
                            failed_count = len(round_attempts) - passed_count

                            round_title = f"Round {round_num}" if round_num == 0 else f"Correction Round {round_num}"
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**{round_title}**")
                            with col2:
                                st.markdown(f"✅ {passed_count}/{len(round_attempts)} passed")
                            with col3:
                                st.markdown(f"❌ {failed_count}/{len(round_attempts)} failed")

                            st.markdown("---")

                            # Display individual samples
                            for idx, attempt in enumerate(round_attempts):
                                data = attempt.get('data', {})
                                comp_result = data.get('compilation_result', {})
                                passed = comp_result.get('pass', False)
                                complete = comp_result.get('complete', False)

                                # Status is ✅ only if BOTH pass AND complete (no sorry)
                                status_emoji = "✅" if (passed and complete) else "❌"
                                sample_name = data.get('name', f'Sample {idx}')

                                # Count axioms used in this proof (try multiple code field names)
                                code = data.get('full_code', '') or data.get('code', '') or data.get('lean4_code', '')
                                defined_axioms, used_axioms, used_axiom_names = count_axioms_in_code(code)

                                expander_title = f"{status_emoji} {sample_name} ({used_axioms}/{defined_axioms} axioms used)"

                                with st.expander(expander_title):
                                    # Show compilation status
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**Compiled:** {'✅ Yes' if passed else '❌ No'}")
                                    with col2:
                                        complete = comp_result.get('complete', False)
                                        st.markdown(f"**Proof Complete:** {'✅ Yes' if complete else '❌ No (has sorry)'}")

                                    # Show used axioms
                                    if used_axiom_names:
                                        axioms_list = ", ".join(sorted(used_axiom_names))
                                        st.markdown(f"**Used Axioms:** {axioms_list}")

                                    # Show prompt, reasoning summary, and compilation summary using shared function
                                    render_proof_attempt_expandables(data, comp_result)

                                    # Show model reasoning if available
                                    model_output = data.get('model_output')
                                    if model_output:
                                        with st.expander("💭 Model Reasoning"):
                                            st.markdown(model_output)

                                    # Show code (expandable)
                                    if code:
                                        with st.expander("📄 Code"):
                                            st.code(code, language="lean")

                                    # Show errors in expandable
                                    errors = comp_result.get('errors', [])
                                    if errors:
                                        with st.expander(f"❌ Compilation Errors ({len(errors)})"):
                                            for error in errors:
                                                if isinstance(error, dict):
                                                    st.text(error.get('data', str(error)))
                                                else:
                                                    st.text(str(error))


def _handle_load_attempts_lemma(breakdown: Breakdown, lemma_id: int):
    """
    Handle the load attempts button click for a lemma.
    Loads full proof attempts from hierarchical dump into the session.

    Args:
        breakdown: Breakdown object to load attempts for
        lemma_id: Lemma ID to load attempts for
    """
    try:
        # Get hierarchical_dir from session state
        hierarchical_dir = st.session_state.get('hierarchical_dir')
        if not hierarchical_dir:
            st.error("❌ Hierarchical directory path not set in session state.")
            return

        # Get the session object
        session = st.session_state.get('session')
        if not session:
            st.error("❌ Session not loaded. Please load data first.")
            return

        # Check if the hierarchical directory actually exists
        from pathlib import Path
        hierarchical_path = Path(hierarchical_dir)
        if not hierarchical_path.exists():
            st.error(f"❌ Hierarchical directory not found at: {hierarchical_path}")
            return

        # Load the attempts
        with st.spinner(f"Loading lemma {lemma_id} proof attempts..."):
            session.load_attempts(
                origin_problem_id=breakdown.origin_problem_id,
                round_id=breakdown.round_id,
                breakdown_id=breakdown.breakdown_id,
                lemma_id=lemma_id,
                hierarchical_dir=hierarchical_path
            )

        st.success(f"✅ Successfully loaded lemma {lemma_id} proof attempts!")

        # Rerun to refresh the UI with the newly loaded data
        st.rerun()

    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
    except ValueError as e:
        st.error(f"❌ Invalid data: {e}")
    except Exception as e:
        st.error(f"❌ Error loading lemma {lemma_id} attempts: {e}")
        import traceback
        st.error(traceback.format_exc())
