"""
Optimal Decision Tree Learning via SAT Encoding

This module implements exact algorithms for learning optimal binary decision trees
using Boolean Satisfiability (SAT) solving. It supports two optimization objectives:
1. Min-Depth: Find the minimum depth tree that correctly classifies all training data
2. Max-Accuracy: Find the fixed-depth tree that maximizes classification accuracy

The approach encodes the decision tree learning problem as a SAT formula where:
- Each internal node selects exactly one feature for splitting
- Data points are consistently routed through the tree based on feature values
- Each leaf is assigned a single class label
- Classification constraints ensure training accuracy

References:
- Narodytska et al. "Learning Optimal Decision Trees with SAT" (2018)
"""

using Satisfiability
using AbstractTrees
using Random
using DelimitedFiles

# Include external dataset loading utilities
include("loaddataset.jl")

#=============================================================================
    Type Definitions
=============================================================================#

"""
    Node

Represents a node in a binary decision tree.

# Fields
- `t::Int`: Node identifier (BFS numbering: root=1, children of i are 2i and 2i+1)
- `leaf::Bool`: Whether this is a leaf node
- `children::Vector{Node}`: Child nodes (empty for leaves)
"""
struct Node
    t::Int
    leaf::Bool
    children::Vector{Node}
end

# Enable tree traversal for Node
AbstractTrees.children(node::Node) = node.children

"""
    TreeSAT

Represents a decoded decision tree with learned parameters.

# Fields
- `t::Int`: Node identifier
- `leaf::Bool`: Whether this is a leaf node
- `feature::Union{Int,Nothing}`: Index of feature used for splitting (internal nodes only)
- `threshold::Union{Float64,Nothing}`: Split threshold (internal nodes only)
- `label::Union{String,Nothing}`: Class label (leaf nodes only)
- `children::Vector{TreeSAT}`: Child nodes (empty for leaves)
"""
struct TreeSAT
    t::Int
    leaf::Bool
    feature::Union{Int,Nothing}
    threshold::Union{Float64,Nothing}
    label::Union{String,Nothing}
    children::Vector{TreeSAT}
end

# Enable tree traversal for TreeSAT
AbstractTrees.children(node::TreeSAT) = node.children

"""
    DecodedSolution

Contains the decoded parameters of a learned decision tree.

# Fields
- `beta::Dict{Int,Int}`: Maps internal node ID to selected feature index
- `alpha::Dict{Int,Float64}`: Maps internal node ID to split threshold
- `theta::Dict{Int,String}`: Maps leaf node ID to class label
- `accuracy::Float64`: Training accuracy (fraction of correctly classified points)
"""
struct DecodedSolution
    beta::Dict{Int,Int}
    alpha::Dict{Int,Float64}
    theta::Dict{Int,String}
    accuracy::Float64
end

#=============================================================================
    Tree Construction
=============================================================================#

"""
    complete_tree!(tree::Dict{Int,Node}, max_depth::Int) -> Dict{Int,Node}

Create a complete binary tree of specified depth with BFS numbering.

Nodes are numbered in breadth-first order:
- Root is node 1
- Children of node i are nodes 2i (left) and 2i+1 (right)
- Leaves are at depth `max_depth` with IDs from 2^max_depth to 2^(max_depth+1)-1

# Arguments
- `tree::Dict{Int,Node}`: Dictionary to populate with nodes (modified in-place)
- `max_depth::Int`: Depth of the tree (root has depth 0)

# Returns
- `Dict{Int,Node}`: The populated tree dictionary
"""
function complete_tree!(tree::Dict{Int,Node}, max_depth::Int)::Dict{Int,Node}
    # Create leaf nodes at the deepest level
    leaf_start = 2^max_depth
    leaf_end = 2^(max_depth + 1) - 1

    for node_id in leaf_start:leaf_end
        tree[node_id] = Node(node_id, true, Node[])
    end

    # Create internal nodes from bottom to top
    for node_id in (leaf_start - 1):-1:1
        left_child = tree[2 * node_id]
        right_child = tree[2 * node_id + 1]
        tree[node_id] = Node(node_id, false, [left_child, right_child])
    end

    return tree
end

#=============================================================================
    Ancestor Computation
=============================================================================#

"""
    compute_left_ancestors(tree::Dict{Int,Node}, leaves::Vector{Node}) -> Dict{Int,Vector{Int}}

Compute left ancestors for each leaf node.

For each leaf t, Al[t] contains all ancestor nodes from which t is reachable
via their left branch.

# Arguments
- `tree::Dict{Int,Node}`: The complete binary tree
- `leaves::Vector{Node}`: Vector of leaf nodes

# Returns
- `Dict{Int,Vector{Int}}`: Maps leaf ID to vector of left ancestor IDs
"""
function compute_left_ancestors(
    tree::Dict{Int,Node},
    leaves::Vector{Node}
)::Dict{Int,Vector{Int}}
    left_ancestors = Dict{Int,Vector{Int}}()

    for leaf in leaves
        left_ancestors[leaf.t] = Int[]
        current_id = leaf.t

        # Traverse up to the root
        while current_id > 1
            parent_id = div(current_id, 2)

            # Check if current is the left child of parent
            if current_id == 2 * parent_id
                push!(left_ancestors[leaf.t], parent_id)
            end

            current_id = parent_id
        end
    end

    return left_ancestors
end

"""
    compute_right_ancestors(tree::Dict{Int,Node}, leaves::Vector{Node}) -> Dict{Int,Vector{Int}}

Compute right ancestors for each leaf node.

For each leaf t, Ar[t] contains all ancestor nodes from which t is reachable
via their right branch.

# Arguments
- `tree::Dict{Int,Node}`: The complete binary tree
- `leaves::Vector{Node}`: Vector of leaf nodes

# Returns
- `Dict{Int,Vector{Int}}`: Maps leaf ID to vector of right ancestor IDs
"""
function compute_right_ancestors(
    tree::Dict{Int,Node},
    leaves::Vector{Node}
)::Dict{Int,Vector{Int}}
    right_ancestors = Dict{Int,Vector{Int}}()

    for leaf in leaves
        right_ancestors[leaf.t] = Int[]
        current_id = leaf.t

        # Traverse up to the root
        while current_id > 1
            parent_id = div(current_id, 2)

            # Check if current is the right child of parent
            if current_id == 2 * parent_id + 1
                push!(right_ancestors[leaf.t], parent_id)
            end

            current_id = parent_id
        end
    end

    return right_ancestors
end

#=============================================================================
    SAT Encoding - Min-Depth Problem
=============================================================================#

"""
    solve_min_depth_optimal_dt(
        features_train::Matrix{T},
        labels_train::Vector{String},
        max_depth_limit::Int=10
    ) where T<:Real -> Union{DecodedSolution,Nothing}

Find the minimum depth decision tree that correctly classifies all training data.

Uses iterative deepening: tries depth 1, 2, ..., max_depth_limit until finding
a satisfying tree. The SAT encoding ensures:
- Each internal node tests exactly one feature
- Data routing respects feature orderings
- Each leaf has exactly one label
- All training points are correctly classified

# Arguments
- `features_train::Matrix{T}`: Training features (n_samples × n_features)
- `labels_train::Vector{String}`: Training labels (n_samples)
- `max_depth_limit::Int`: Maximum tree depth to try (default: 10)

# Returns
- `DecodedSolution`: The learned tree parameters, or `nothing` if no solution found
"""
function solve_min_depth_optimal_dt(
    features_train::Matrix{T},
    labels_train::Vector{String},
    max_depth_limit::Int = 10;
    silent::Bool=false
)::Union{DecodedSolution,Nothing} where {T<:Real}

    n_samples, n_features = size(features_train)
    unique_labels = unique(labels_train)
    n_classes = length(unique_labels)

    # Create bidirectional mapping between labels and integer indices
    label_to_idx = Dict(label => idx for (idx, label) in enumerate(unique_labels))
    idx_to_label = Dict(idx => label for (idx, label) in enumerate(unique_labels))

    # Try increasing depths until solution found
    for depth in 1:max_depth_limit
        silent|| println("Attempting depth: $depth")

        # Build complete binary tree
        tree = Dict{Int,Node}()
        complete_tree!(tree, depth)
        leaves = [node for (_, node) in tree if node.leaf]

        # Compute ancestor sets for routing constraints
        Al = compute_left_ancestors(tree, leaves)
        Ar = compute_right_ancestors(tree, leaves)

        # Collect node IDs
        internal_ids = [node.t for (_, node) in tree if !node.leaf]
        leaf_ids = [leaf.t for leaf in leaves]

        # Create SAT variables
        # a[t,j]: node t uses feature j for splitting
        @satvariable(a[internal_ids, 1:n_features], Bool)

        # s[i,t]: sample i goes left at node t
        @satvariable(s[1:n_samples, internal_ids], Bool)

        # z[i,l]: sample i reaches leaf l
        @satvariable(z[1:n_samples, leaf_ids], Bool)

        # g[l,c]: leaf l has class label c
        @satvariable(g[leaf_ids, 1:n_classes], Bool)

        # Build SAT clauses
        clauses = BoolExpr[]

        # === CONSTRAINTS FOR INTERNAL NODES ===
        for node_t in internal_ids
            # Constraint 1 & 2: Exactly one feature per node
            feature_vars = [a[node_t, j] for j in 1:n_features]

            # At least one feature
            push!(clauses, reduce(∨, feature_vars))

            # At most one feature (pairwise exclusion)
            for i in 1:n_features, j in (i+1):n_features
                push!(clauses, ¬feature_vars[i] ∨ ¬feature_vars[j])
            end

            # Feature-specific routing constraints
            for j in 1:n_features
                # Sort samples by feature j values
                sorted_indices = sortperm(view(features_train, :, j))
                ba = a[node_t, j]

                # Constraint 3 & 4: Consistent ordering
                for i in 2:length(sorted_indices)
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    bs_curr = s[curr_idx, node_t]
                    bs_prev = s[prev_idx, node_t]

                    if features_train[prev_idx, j] != features_train[curr_idx, j]
                        # Different values: prev goes left => curr goes left
                        # ¬a[t,j] ∨ s[prev,t] ∨ ¬s[curr,t]
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    else
                        # Equal values: must go in the same direction
                        # ¬a[t,j] ∨ (s[prev,t] ↔ s[curr,t])
                        push!(clauses, ¬ba ∨ ¬bs_prev ∨ bs_curr)
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    end
                end

                # Constraint 9: Smallest value goes left
                if !isempty(sorted_indices)
                    first_idx = sorted_indices[1]
                    push!(clauses, ¬ba ∨ s[first_idx, node_t])

                    # Constraint 10: Largest value goes right
                    last_idx = sorted_indices[end]
                    push!(clauses, ¬ba ∨ ¬s[last_idx, node_t])
                end
            end
        end

        # === CONSTRAINTS FOR LEAVES ===
        for (leaf_idx, leaf_t) in enumerate(leaf_ids)
            # Convert absolute leaf ID to relative index (1-based)
            ll = leaf_idx

            # Constraint 5: Left path consistency
            # If sample reaches this leaf, it must go left at all left ancestors
            for ancestor_t in Al[leaf_t]
                for sample_idx in 1:n_samples
                    # ¬z[i,l] ∨ s[i,ancestor]
                    push!(clauses, ¬z[sample_idx, ll] ∨ s[sample_idx, ancestor_t])
                end
            end

            # Constraint 6: Right path consistency
            # If sample reaches this leaf, it must go right at all right ancestors
            for ancestor_t in Ar[leaf_t]
                for sample_idx in 1:n_samples
                    # ¬z[i,l] ∨ ¬s[i,ancestor]
                    push!(clauses, ¬z[sample_idx, ll] ∨ ¬s[sample_idx, ancestor_t])
                end
            end

            # Constraint 7: Path completeness
            # If sample doesn't reach this leaf, it violates at least one ancestor constraint
            for sample_idx in 1:n_samples
                violations = BoolExpr[]

                # Collect violations: going right at left ancestors or left at right ancestors
                for left_anc in Al[leaf_t]
                    push!(violations, ¬s[sample_idx, left_anc])
                end
                for right_anc in Ar[leaf_t]
                    push!(violations, s[sample_idx, right_anc])
                end

                if !isempty(violations)
                    # z[i,l] ∨ (at least one violation)
                    push!(clauses, z[sample_idx, ll] ∨ reduce(∨, violations))
                end
            end

            # Constraint 8: At most one label per leaf
            label_vars = [g[ll, c_idx] for c_idx in 1:n_classes]
            for i in 1:n_classes, j in (i+1):n_classes
                push!(clauses, ¬label_vars[i] ∨ ¬label_vars[j])
            end

            # Constraint 11: Correct classification (Min-Depth specific)
            # If sample i reaches leaf l, l must have sample i's correct label
            for sample_idx in 1:n_samples
                correct_label = labels_train[sample_idx]
                correct_idx = label_to_idx[correct_label]
                # ¬z[i,l] ∨ g[l,correct_class]
                push!(clauses, ¬z[sample_idx, ll] ∨ g[ll, correct_idx])
            end
        end

        # === GLOBAL CONSTRAINTS ===
        # Each sample reaches exactly one leaf
        for sample_idx in 1:n_samples
            leaf_vars = [z[sample_idx, l] for l in 1:length(leaf_ids)]

            # At least one leaf
            push!(clauses, reduce(∨, leaf_vars))

            # At most one leaf
            for i in 1:length(leaf_vars), j in (i+1):length(leaf_vars)
                push!(clauses, ¬leaf_vars[i] ∨ ¬leaf_vars[j])
            end
        end

        # Solve SAT instance
        silent|| println("Solving SAT with $(length(clauses)) clauses...")
        formula = reduce(∧, clauses)

        try
            status = sat!(formula, solver = Z3())

            if status == :SAT
                silent|| println("Solution found at depth $depth")
                return decode_solution(
                    tree, a, s, z, g,
                    features_train, labels_train,
                    Al, Ar,
                    label_to_idx, idx_to_label
                )
            else
                silent|| println("No solution at depth $depth")
            end
        catch e
            silent|| println("Error during SAT solving: $e")
            return nothing
        end
    end

    silent|| println("No solution found up to depth $max_depth_limit")
    return nothing
end

#=============================================================================
    Solution Decoding
=============================================================================#

"""
    decode_solution(
        tree::Dict{Int,Node},
        a, s, z, g,
        features_train::Matrix{T},
        labels_train::Vector{String},
        Al::Dict{Int,Vector{Int}},
        Ar::Dict{Int,Vector{Int}},
        label_to_idx::Dict{String,Int},
        idx_to_label::Dict{Int,String}
    ) where T<:Real -> DecodedSolution

Extract human-readable decision tree parameters from SAT solution.

Decodes the Boolean variable assignments into:
- β (beta): Feature selection for each internal node
- α (alpha): Split threshold for each internal node
- θ (theta): Class label for each leaf node

# Arguments
- `tree::Dict{Int,Node}`: The tree structure
- `a, s, z, g`: SAT variable assignments
- `features_train`: Training features
- `labels_train`: Training labels
- `Al, Ar`: Ancestor sets
- `label_to_idx, idx_to_label`: Label mappings

# Returns
- `DecodedSolution`: Decoded tree parameters with accuracy
"""
function decode_solution(
    tree::Dict{Int,Node},
    a, s, z, g,
    features_train::Matrix{T},
    labels_train::Vector{String},
    Al::Dict{Int,Vector{Int}},
    Ar::Dict{Int,Vector{Int}},
    label_to_idx::Dict{String,Int},
    idx_to_label::Dict{Int,String};
    silent::Bool=false
)::DecodedSolution where {T<:Real}

    silent|| println("\n" * "="^60)
    silent|| println("DECODING SAT SOLUTION")
    silent|| println("="^60)

    n_samples, n_features = size(features_train)
    internal_ids = [node.t for (_, node) in tree if !node.leaf]
    leaf_ids = [node.t for (_, node) in tree if node.leaf]
    n_classes = length(idx_to_label)

    # Decode β: Feature selection
    beta = Dict{Int,Int}()
    silent|| println("\n📊 Feature Selection (β):")
    for node_t in internal_ids
        for j in 1:n_features
            if value(a[node_t, j]) == true
                beta[node_t] = j
                silent|| println("  Node $node_t → Feature $j")
                break
            end
        end
    end

    # Decode α: Split thresholds
    alpha = Dict{Int,Float64}()
    silent|| println("\n🔍 Split Thresholds (α):")
    for node_t in internal_ids
        if haskey(beta, node_t)
            feature_j = beta[node_t]
            sorted_indices = sortperm(view(features_train, :, feature_j))

            # Find the split point between left and right samples
            split_found = false
            for i in 2:length(sorted_indices)
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]

                # Check if split occurs between these samples
                if value(s[prev_idx, node_t]) == true &&
                   value(s[curr_idx, node_t]) == false

                    # Threshold is midpoint between the two values
                    alpha[node_t] = (
                        features_train[prev_idx, feature_j] +
                        features_train[curr_idx, feature_j]
                    ) / 2.0

                    silent|| println("  Node $node_t → Threshold $(alpha[node_t])")
                    split_found = true
                    break
                end
            end

            # Fallback: use maximum left value
            if !split_found
                max_left_value = -Inf
                for idx in sorted_indices
                    if value(s[idx, node_t]) == true
                        max_left_value = max(max_left_value, features_train[idx, feature_j])
                    end
                end

                if max_left_value != -Inf
                    alpha[node_t] = max_left_value
                    silent|| println("  Node $node_t → Threshold $(alpha[node_t]) (fallback)")
                end
            end
        end
    end

    # Decode θ: Leaf labels
    theta = Dict{Int,String}()
    silent|| println("\n Leaf Labels (θ):")
    for (leaf_idx, leaf_t) in enumerate(leaf_ids)
        for c_idx in 1:n_classes
            if value(g[leaf_idx, c_idx]) == true
                label_str = idx_to_label[c_idx]
                theta[leaf_t] = label_str
                silent|| println("  Leaf $leaf_idx (node $leaf_t) → Label \"$label_str\"")
                break
            end
        end
    end

    # Verify classification accuracy
    correct = 0
    silent|| println("\n Sample Classification:")
    for sample_idx in 1:n_samples
        for (leaf_idx, leaf_t) in enumerate(leaf_ids)
            if value(z[sample_idx, leaf_idx]) == true
                predicted = get(theta, leaf_t, "UNKNOWN")
                actual = labels_train[sample_idx]
                is_correct = predicted == actual

                if is_correct
                    correct += 1
                end

                status_symbol = is_correct ? "✓" : "✗"
                silent|| println("  $status_symbol Sample $sample_idx: predicted=\"$predicted\", actual=\"$actual\"")
                break
            end
        end
    end

    accuracy = correct / n_samples
    silent|| println("\n" * "="^60)
    silent|| println("Training Accuracy: $correct/$n_samples = $(round(accuracy * 100, digits=2))%")
    silent|| println("="^60)

    return DecodedSolution(beta, alpha, theta, accuracy)
end

#=============================================================================
    Tree Visualization
=============================================================================#

"""
    build_tree_sat(
        alpha::Dict{Int,Float64},
        beta::Dict{Int,Int},
        theta::Dict{Int,String}
    ) -> Dict{Int,TreeSAT}

Construct a TreeSAT structure from decoded solution parameters.

# Arguments
- `alpha::Dict{Int,Float64}`: Split thresholds for internal nodes
- `beta::Dict{Int,Int}`: Feature selections for internal nodes
- `theta::Dict{Int,String}`: Class labels for leaves

# Returns
- `Dict{Int,TreeSAT}`: Dictionary mapping node IDs to TreeSAT nodes
"""
function build_tree_sat(
    alpha::Dict{Int,Float64},
    beta::Dict{Int,Int},
    theta::Dict{Int,String}
)::Dict{Int,TreeSAT}

    # Find maximum node ID to determine depth
    max_node = maximum(union(keys(alpha), keys(beta), keys(theta)))
    max_depth = floor(Int, log2(max_node + 1)) - 1

    nodes = Dict{Int,TreeSAT}()

    # Create leaf nodes (deepest level)
    leaf_start = 2^max_depth
    leaf_end = 2^(max_depth + 1) - 1

    for node_id in leaf_start:leaf_end
        label = get(theta, node_id, "UNKNOWN")
        nodes[node_id] = TreeSAT(
            node_id,
            true,           # is_leaf
            nothing,        # feature
            nothing,        # threshold
            label,          # label
            TreeSAT[]      # children
        )
    end

    # Create internal nodes bottom-up
    for node_id in (leaf_start - 1):-1:1
        left_child = nodes[2 * node_id]
        right_child = nodes[2 * node_id + 1]

        feature = get(beta, node_id, nothing)
        threshold = get(alpha, node_id, nothing)

        nodes[node_id] = TreeSAT(
            node_id,
            false,          # is_leaf
            feature,        # feature
            threshold,      # threshold
            nothing,        # label
            [left_child, right_child]
        )
    end

    return nodes
end

#=============================================================================
    Test Dataset Generation
=============================================================================#

"""
    create_simple_dummy_dataset_depth5() -> Tuple{Matrix{Int},Vector{String}}

Generate a synthetic binary feature dataset requiring depth 5 for perfect classification.

Creates a 32-sample dataset with 5 binary features and 3 classes (A, B, C).
The dataset is designed to be separable only with a sufficiently deep tree.

# Returns
- `Tuple{Matrix{Int},Vector{String}}`: (features, labels) where features is 32×5
"""
function create_simple_dummy_dataset_depth5()::Tuple{Matrix{Int},Vector{String}}
    # 32 samples with 5 binary features each
    features = [
        [0, 0, 0, 0, 0],  # Sample 1: Class A
        [0, 0, 0, 0, 1],  # Sample 2: Class B
        [0, 0, 0, 1, 0],  # Sample 3: Class A
        [0, 0, 0, 1, 1],  # Sample 4: Class C
        [0, 0, 1, 0, 0],  # Sample 5: Class B
        [0, 0, 1, 0, 1],  # Sample 6: Class A
        [0, 0, 1, 1, 0],  # Sample 7: Class C
        [0, 0, 1, 1, 1],  # Sample 8: Class B
        [0, 1, 0, 0, 0],  # Sample 9: Class C
        [0, 1, 0, 0, 1],  # Sample 10: Class A
        [0, 1, 0, 1, 0],  # Sample 11: Class B
        [0, 1, 0, 1, 1],  # Sample 12: Class C
        [0, 1, 1, 0, 0],  # Sample 13: Class A
        [0, 1, 1, 0, 1],  # Sample 14: Class B
        [0, 1, 1, 1, 0],  # Sample 15: Class A
        [0, 1, 1, 1, 1],  # Sample 16: Class C
        [1, 0, 0, 0, 0],  # Sample 17: Class B
        [1, 0, 0, 0, 1],  # Sample 18: Class C
        [1, 0, 0, 1, 0],  # Sample 19: Class A
        [1, 0, 0, 1, 1],  # Sample 20: Class B
        [1, 0, 1, 0, 0],  # Sample 21: Class C
        [1, 0, 1, 0, 1],  # Sample 22: Class B
        [1, 0, 1, 1, 0],  # Sample 23: Class A
        [1, 0, 1, 1, 1],  # Sample 24: Class C
        [1, 1, 0, 0, 0],  # Sample 25: Class A
        [1, 1, 0, 0, 1],  # Sample 26: Class B
        [1, 1, 0, 1, 0],  # Sample 27: Class C
        [1, 1, 0, 1, 1],  # Sample 28: Class A
        [1, 1, 1, 0, 0],  # Sample 29: Class B
        [1, 1, 1, 0, 1],  # Sample 30: Class C
        [1, 1, 1, 1, 0],  # Sample 31: Class B
        [1, 1, 1, 1, 1],  # Sample 32: Class A
    ]

    labels = [
        "A", "B", "A", "C", "B", "A", "C", "B",
        "C", "A", "B", "C", "A", "B", "A", "C",
        "B", "C", "A", "B", "C", "B", "A", "C",
        "A", "B", "C", "A", "B", "C", "B", "A",
    ]

    # Convert to matrix format
    n_samples = length(features)
    n_features = length(features[1])
    features_matrix = zeros(Int, n_samples, n_features)

    for i in 1:n_samples
        features_matrix[i, :] = features[i]
    end

    return features_matrix, labels
end

#=============================================================================
    Main Execution
=============================================================================#

"""
    main()

Main entry point for the optimal decision tree learning demo.

Loads the synthetic dataset and attempts to find the minimum depth tree
that correctly classifies all training samples.
"""
function main(;silent::Bool=false)
    silent|| println("\n" * "="^60)
    silent|| println("OPTIMAL DECISION TREE LEARNING VIA SAT")
    silent|| println("="^60)

    # Load training data
    features_train, labels_train = create_simple_dummy_dataset_depth5()

    n_samples, n_features = size(features_train)
    n_classes = length(unique(labels_train))

    silent|| println("\nDataset Loaded:")
    silent|| println("  Samples: $n_samples")
    silent|| println("  Features: $n_features")
    silent|| println("  Classes: $n_classes ($(unique(labels_train)))")

    # Solve Min-Depth problem
    silent|| println("\nSearching for optimal decision tree...")
    solution = solve_min_depth_optimal_dt(features_train, labels_train, 10)

    if solution !== nothing
        silent|| println("\nOptimal tree found!")

        # Build and visualize the tree structure
        tree_sat = build_tree_sat(solution.alpha, solution.beta, solution.theta)

        silent|| println("\nTree Structure:")
        print_tree(tree_sat[1])
    else
        silent|| println("\nNo solution found within depth limit")
    end
end

#=============================================================================
    Max-Accuracy Problem (TODO)
=============================================================================#

"""
    solve_max_accuracy_optimal_dt(
        features_train::Matrix{T},
        labels_train::Vector{String},
        target_depth::Int
    ) where T<:Real -> Union{DecodedSolution,Nothing}

Find the fixed-depth decision tree that maximizes classification accuracy.

This is the dual problem to Min-Depth: instead of finding the minimum depth
for perfect classification, we fix the depth and maximize the number of
correctly classified samples using MaxSAT.

# Implementation Strategy
1. Use the same hard constraints as Min-Depth (constraints 1-10)
2. Remove the hard classification constraint (constraint 11)
3. Add soft clauses: maximize the number of samples with p[i] = true
4. p[i] = true iff sample i is correctly classified

# Arguments
- `features_train::Matrix{T}`: Training features (n_samples × n_features)
- `labels_train::Vector{String}`: Training labels (n_samples)
- `target_depth::Int`: Fixed tree depth to use

# Returns
- `DecodedSolution`: The learned tree parameters, or `nothing` if solving fails

# Note
This function is currently a stub. Implementation requires:
1. A MaxSAT solver (e.g., RC2, MaxHS)
2. Separating hard constraints from soft optimization objectives
3. Decoding the solution similar to Min-Depth

# Example MaxSAT Encoding
```julia
# Hard constraints (structure + routing)
hard_clauses = [
    # Constraints 1-10 from Min-Depth
    # ...
]

# Soft constraints (maximize correct classifications)
soft_clauses = [p[i] for i in 1:n_samples]

# Constraint 12: p[i] is true iff sample i is correctly classified
# For each sample i and each leaf l:
#   ¬p[i] ∨ ¬z[i,l] ∨ g[l,correct_label[i]]
```
"""
function solve_max_accuracy_optimal_dt(
    features_train::Matrix{T},
    labels_train::Vector{String},
    target_depth::Int;
    silent::Bool=false
)::Union{DecodedSolution,Nothing} where {T<:Real}

    n_samples, n_features = size(features_train)
    unique_labels = unique(labels_train)
    n_classes = length(unique_labels)

    silent|| println("Max-Accuracy solver for depth $target_depth")
    silent|| println("This feature is not yet implemented.")
    silent|| println("\nRequired components:")
    silent|| println("  1. MaxSAT solver integration (e.g., RC2, MaxHS)")
    silent|| println("  2. Hard constraints: tree structure + routing (constraints 1-10)")
    silent|| println("  3. Soft constraints: maximize correct classifications")
    silent|| println("  4. Constraint 12: p[i] ↔ (sample i correctly classified)")

    # Build tree structure
    tree = Dict{Int,Node}()
    complete_tree!(tree, target_depth)
    leaves = [node for (_, node) in tree if node.leaf]

    # Compute ancestors
    Al = compute_left_ancestors(tree, leaves)
    Ar = compute_right_ancestors(tree, leaves)

    # Create mappings
    label_to_idx = Dict(label => idx for (idx, label) in enumerate(unique_labels))
    idx_to_label = Dict(idx => label for (idx, label) in enumerate(unique_labels))

    # Collect node IDs
    internal_ids = [node.t for (_, node) in tree if !node.leaf]
    leaf_ids = [leaf.t for leaf in leaves]

    # TODO: Create SAT variables including p[i] for correct classification
    # @satvariable(a[internal_ids, 1:n_features], Bool)
    # @satvariable(s[1:n_samples, internal_ids], Bool)
    # @satvariable(z[1:n_samples, leaf_ids], Bool)
    # @satvariable(g[leaf_ids, 1:n_classes], Bool)
    # @satvariable(p[1:n_samples], Bool)  # New: correctness indicators

    # TODO: Build hard constraints (same as Min-Depth constraints 1-10)
    # hard_clauses = BoolExpr[]
    # ... (add all structural and routing constraints)

    # TODO: Add Constraint 12 to hard clauses
    # For each sample i:
    #   For each leaf l:
    #     For correct class c of sample i:
    #       hard_clauses.push(¬p[i] ∨ ¬z[i,l] ∨ g[l,c])

    # TODO: Define soft clauses (maximize correct classifications)
    # soft_clauses = [p[i] for i in 1:n_samples]

    # TODO: Solve with MaxSAT solver
    # solution = maxsat!(hard_clauses, soft_clauses, solver=...)

    silent|| println("\nReturning nothing (not implemented)")
    return nothing
end

#=============================================================================
    Utility Functions
=============================================================================#

"""
    print_tree_info(node::TreeSAT, indent::String="", is_last::Bool=true)

Pretty-print a TreeSAT structure with tree visualization characters.

# Arguments
- `node::TreeSAT`: The node to print
- `indent::String`: Current indentation string
- `is_last::Bool`: Whether this is the last child of its parent
"""
function print_tree_info(node::TreeSAT, indent::String="", is_last::Bool=true;silent::Bool=false)
    # Choose the appropriate tree characters
    connector = is_last ? "└── " : "├── "

    # Print current node
    if node.leaf
        silent|| println(indent * connector * "Leaf $(node.t): $(node.label)")
    else
        feature_str = node.feature !== nothing ? "Feature $(node.feature)" : "No feature"
        threshold_str = node.threshold !== nothing ?
            "Threshold $(round(node.threshold, digits=3))" : "No threshold"
        silent|| println(indent * connector * "Node $(node.t): $feature_str, $threshold_str")

        # Prepare indent for children
        child_indent = indent * (is_last ? "    " : "│   ")

        # Print children
        if length(node.children) > 0
            for (i, child) in enumerate(node.children)
                print_tree_info(child, child_indent, i == length(node.children))
            end
        end
    end
end

"""
    validate_tree_consistency(
        tree::Dict{Int,TreeSAT},
        features::Matrix{T},
        labels::Vector{String}
    ) where T<:Real -> Bool

Validate that the learned tree correctly classifies all training samples.

# Arguments
- `tree::Dict{Int,TreeSAT}`: The learned tree
- `features::Matrix{T}`: Training features
- `labels::Vector{String}`: Training labels

# Returns
- `Bool`: True if all samples are correctly classified
"""
function validate_tree_consistency(
    tree::Dict{Int,TreeSAT},
    features::Matrix{T},
    labels::Vector{String}
)::Bool where {T<:Real}

    n_samples = size(features, 1)
    all_correct = true

    silent|| println("\n" * "="^60)
    silent|| println("TREE VALIDATION")
    silent|| println("="^60)

    for sample_idx in 1:n_samples
        # Traverse tree to find prediction
        current_node = tree[1]
        path = [1]

        while !current_node.leaf
            feature_idx = current_node.feature
            threshold = current_node.threshold

            if feature_idx === nothing || threshold === nothing
                silent|| println("   Warning: Node $(current_node.t) missing parameters")
                all_correct = false
                break
            end

            # Decide which child to follow
            if features[sample_idx, feature_idx] <= threshold
                current_node = current_node.children[1]  # Go left
            else
                current_node = current_node.children[2]  # Go right
            end

            push!(path, current_node.t)
        end

        # Check prediction
        predicted = current_node.label
        actual = labels[sample_idx]
        is_correct = predicted == actual

        if !is_correct
            all_correct = false
        end

        status = is_correct ? "✓" : "✗"
        path_str = join(path, " → ")
        silent|| println("$status Sample $sample_idx: $path_str → Predicted: $predicted, Actual: $actual")
    end

    silent|| println("="^60)
    if all_correct
        silent|| println("   All samples correctly classified!")
    else
        silent|| println("   Some samples misclassified")
    end
    silent|| println("="^60)

    return all_correct
end

"""
    export_tree_to_dot(
        tree::Dict{Int,TreeSAT},
        filename::String="tree.dot"
    )

Export the tree structure to Graphviz DOT format for visualization.

# Arguments
- `tree::Dict{Int,TreeSAT}`: The tree to export
- `filename::String`: Output filename (default: "tree.dot")

# Usage
After generating the .dot file, render it with:
```bash
dot -Tpng tree.dot -o tree.png
```
"""
function export_tree_to_dot(
    tree::Dict{Int,TreeSAT},
    filename::String="tree.dot";
    silent::Bool=false
)
    open(filename, "w") do io
        silent|| println(io, "digraph DecisionTree {")
        silent|| println(io, "    node [shape=box, style=rounded];")
        silent|| println(io, "    edge [fontsize=10];")

        for (node_id, node) in tree
            if node.leaf
                # Leaf node (green background)
                label = "Leaf $(node.t)\\nClass: $(node.label)"
                silent|| println(io, "    node$node_id [label=\"$label\", fillcolor=lightgreen, style=filled];")
            else
                # Internal node (blue background)
                feature_str = node.feature !== nothing ? "F$(node.feature)" : "?"
                threshold_str = node.threshold !== nothing ?
                    "$(round(node.threshold, digits=2))" : "?"
                label = "Node $(node.t)\\n$feature_str ≤ $threshold_str"
                silent|| println(io, "    node$node_id [label=\"$label\", fillcolor=lightblue, style=filled];")

                # Add edges to children
                if length(node.children) >= 2
                    left_child = node.children[1]
                    right_child = node.children[2]
                    silent|| println(io, "    node$node_id -> node$(left_child.t) [label=\"True\"];")
                    silent|| println(io, "    node$node_id -> node$(right_child.t) [label=\"False\"];")
                end
            end
        end

        silent|| println(io, "}")
    end

    silent|| println("Tree exported to $filename")
    silent|| println("To visualize: dot -Tpng $filename -o tree.png")
end

#=============================================================================
    Entry Point
=============================================================================#

# Execute main function if running as script

main()
