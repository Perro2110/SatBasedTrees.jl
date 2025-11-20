using Satisfiability
using AbstractTrees
using Random
using DelimitedFiles
using DecisionTree
using StatsBase
using SoleDecisionTreeInterface

import DecisionTree: Node, Leaf, print_tree
import Satisfiability: ∨, ∧, ¬, sat!, @satvariable, BoolExpr, AbstractExpr, value

include("loaddataset.jl")

#=============================================================================
    SAT-BASED OPTIMAL DECISION TREE LEARNING

    This module implements exact decision tree learning using Boolean
    Satisfiability (SAT) solvers. It formulates the decision tree construction
    as a SAT problem and finds optimal trees according to two criteria:

    1. Min-Depth: Find the shallowest tree that perfectly classifies training data
    2. Max-Accuracy: Find the most accurate tree for a given depth (TODO)

    The approach is based on the paper:
    "Learning Optimal Decision Trees with SAT" (Narodytska et al., 2018)

    Key advantages:
    - Guarantees optimal solutions (no heuristics)
    - Can handle arbitrary splitting criteria
    - Provides interpretable models

    Key limitations:
    - Computationally expensive for large datasets
    - Best suited for small to medium datasets
=============================================================================#

#=============================================================================
    Type Definitions
=============================================================================#

"""
    NodeT

Represents a node in a binary decision tree with BFS numbering.

# Fields
- `t::Integer`: Node identifier using breadth-first search numbering
    - Root node has id = 1
    - Children of node i have ids 2i (left) and 2i+1 (right)
- `leaf::Bool`: Whether this is a leaf node (true) or internal node (false)
- `children::Vector{NodeT}`: Vector of child nodes (empty for leaf nodes)

# BFS Numbering Example
```
        1          <- root (depth 0)
       / \\
      2   3        <- depth 1
     / \\ / \\
    4  5 6  7      <- depth 2 (leaves if max_depth=2)
```

# Notes
- BFS numbering simplifies tree navigation and parent-child relationships
- Parent of node i is always floor(i/2)
- Left child of node i is 2i, right child is 2i+1
- Leaves at depth d have ids in range [2^d, 2^(d+1)-1]
"""
struct NodeT
    t::Integer
    leaf::Bool
    children::Vector{NodeT}
end

# Enable tree traversal using AbstractTrees interface
AbstractTrees.children(node::NodeT) = node.children


"""
    completeTree(t::Dict{Int,NodeT}, max_depth::Int)

Create a complete binary tree of specified depth with BFS numbering.

# Arguments
- `t::Dict{Int,NodeT}`: Empty dictionary to be populated with tree nodes
- `max_depth::Int`: Maximum depth of the tree (root is at depth 0)

# Effects
Populates the dictionary `t` with NodeT objects representing a complete binary
tree where:
- Internal nodes are numbered from 1 to 2^max_depth - 1
- Leaf nodes are numbered from 2^max_depth to 2^(max_depth+1) - 1

# Example
```julia
t = Dict{Int,NodeT}()
completeTree(t, 2)  # Creates tree with 7 nodes: 1 root, 2 internal, 4 leaves

# Tree structure:
#       1          <- internal
#      / \\
#     2   3        <- internal
#    / \\ / \\
#   4  5 6  7      <- leaves
```

# Algorithm
1. Create all leaf nodes first (bottom-up approach)
2. Create internal nodes from bottom to top, linking to their children
3. Each internal node i has children 2i and 2i+1
"""
function completeTree(t::Dict{Int,NodeT}, max_depth::Int)
    # First create all leaf nodes (deepest level)
    for i = (2^max_depth):(2^(max_depth+1)-1)
        t[i] = NodeT(i, true, [])
    end

    # Then create internal nodes from bottom to top
    for i = (2^max_depth-1):-1:1
        t[i] = NodeT(i, false, [t[2*i], t[2*i+1]])
    end
end


"""
    computeLeftAncestors(t::Dict{Int,NodeT}, leaves)

Compute left ancestors for each leaf node in the tree.

# Arguments
- `t::Dict{Int,NodeT}`: Complete tree dictionary
- `leaves`: Collection of leaf nodes

# Returns
- `Dict{Int,Vector{Int}}`: Maps each leaf id to vector of ancestor ids where
  the leaf is reachable via the left branch

# Description
For each leaf, this function identifies all ancestor nodes where taking the
LEFT branch leads toward that leaf. This is crucial for encoding path
constraints in the SAT formulation.

# Example
```julia
# Tree structure:
#       1
#      / \\
#     2   3
#    / \\
#   4   5

Al = computeLeftAncestors(t, leaves)
# Al[4] = [1, 2]  -> Leaf 4 is left child of 2, and 2 is left child of 1
# Al[5] = [2]     -> Leaf 5 is right child of 2, but 2 is left child of 1
# Al[6] = []      -> Leaf 6 is right of 3, and 3 is right of 1
```

# Path Encoding
If a data point reaches leaf t, it must have gone LEFT at all nodes in Al[t].
This is used in Clause 5 of the SAT formulation.
"""
function computeLeftAncestors(t::Dict{Int,NodeT}, leaves)
    Al = Dict{Int,Vector{Int}}()

    for leaf in leaves
        Al[leaf.t] = Int[]
        current = leaf.t

        # Traverse up to the root
        while current > 1
            parent = div(current, 2)
            # If current is the left child of parent
            if current == 2 * parent
                push!(Al[leaf.t], parent)
            end
            current = parent
        end
    end
    return Al
end


"""
    computeRightAncestors(t::Dict{Int,NodeT}, leaves)

Compute right ancestors for each leaf node in the tree.

# Arguments
- `t::Dict{Int,NodeT}`: Complete tree dictionary
- `leaves`: Collection of leaf nodes

# Returns
- `Dict{Int,Vector{Int}}`: Maps each leaf id to vector of ancestor ids where
  the leaf is reachable via the right branch

# Description
For each leaf, this function identifies all ancestor nodes where taking the
RIGHT branch leads toward that leaf. Complementary to computeLeftAncestors.

# Example
```julia
# Tree structure:
#       1
#      / \\
#     2   3
#        / \\
#       6   7

Ar = computeRightAncestors(t, leaves)
# Ar[4] = []      -> Leaf 4: left of 2, left of 1
# Ar[6] = [1, 3]  -> Leaf 6: left of 3, but 3 is right of 1
# Ar[7] = [1, 3]  -> Leaf 7: right of 3, and 3 is right of 1
```

# Path Encoding
If a data point reaches leaf t, it must have gone RIGHT at all nodes in Ar[t].
This is used in Clause 6 of the SAT formulation.
"""
function computeRightAncestors(t::Dict{Int,NodeT}, leaves)
    Ar = Dict{Int,Vector{Int}}()

    for leaf in leaves
        Ar[leaf.t] = Int[]
        current = leaf.t

        # Traverse up to the root
        while current > 1
            parent = div(current, 2)
            # If current is the right child of parent
            if current == 2 * parent + 1
                push!(Ar[leaf.t], parent)
            end
            current = parent
        end
    end
    return Ar
end


#=============================================================================
    Min-Depth Optimal Decision Tree
=============================================================================#

"""
    solveMinDepthOptimalDT(features_train, labels_train, max_depth_limit=10)

Find the minimum-depth decision tree that perfectly classifies the training data.

# Arguments
- `features_train`: Matrix of size (n_samples, n_features) containing training features
- `labels_train`: Vector of length n_samples containing training labels
- `max_depth_limit::Int=10`: Maximum depth to try (prevents infinite search)

# Returns
Named tuple with fields:
- `beta::Dict{Int,Int}`: Maps node id to selected feature index
- `alpha::Dict{Int,Float64}`: Maps node id to splitting threshold
- `theta::Dict{Int,String}`: Maps leaf id to predicted class label
- `accuracy::Float64`: Training accuracy (should be 1.0 for Min-Depth)

Returns `nothing` if no solution found within depth limit.

# Algorithm
1. Try increasing depths starting from 1
2. For each depth, construct a complete binary tree
3. Encode the tree learning problem as a SAT formula
4. Use Z3 solver to find satisfying assignment
5. If SAT, decode solution and return; if UNSAT, try next depth
6. Stop when solution found or max_depth_limit reached

# SAT Variables
- `a[t,j]`: Binary variable indicating if node t uses feature j (β encoding)
- `s[i,t]`: Binary variable indicating if sample i goes LEFT at node t
- `z[i,l]`: Binary variable indicating if sample i reaches leaf l
- `g[l,c]`: Binary variable indicating if leaf l predicts class c (θ encoding)

# SAT Clauses (Constraints)
1. Each internal node selects exactly one feature
2. Feature selection consistency (part of clause 1)
3. Ordering constraint: samples sorted by feature go in consistent directions
4. Equal feature values must go same direction
5. Left path constraint: samples reaching leaf must go left at left ancestors
6. Right path constraint: samples reaching leaf must go right at right ancestors
7. Complete path definition: samples must follow correct path to reach leaf
8. Each leaf predicts at most one class
9. First sorted sample (minimum value) goes left
10. Last sorted sample (maximum value) goes right
11. Correct classification: if sample reaches leaf, leaf must predict correct class

# Example
```julia
# Simple XOR-like dataset
features = [0 0; 0 1; 1 0; 1 1]
labels = ["A", "B", "B", "A"]

solution = solveMinDepthOptimalDT(features, labels, 5)

if solution !== nothing
    println("Found tree at depth with accuracy: ", solution.accuracy)
    # Use beta, alpha, theta to construct decision tree
end
```

# Computational Complexity
- SAT solving is NP-complete, so worst-case exponential
- Practical performance depends on:
  - Number of samples (affects clause count quadratically)
  - Number of features (affects variable count)
  - Depth of optimal tree (affects tree size exponentially)
- Recommended for datasets with < 100 samples

# Notes
- This guarantees finding the shallowest possible tree
- All training samples must be correctly classified
- For noisy data, consider Max-Accuracy variant instead
"""
function solveMinDepthOptimalDT(features_train, labels_train, max_depth_limit = 10)
    max_r, max_c = size(features_train)  # n_samples, n_features
    unique_labels = unique(labels_train)
    n_labels = length(unique_labels)

    # Create bidirectional mapping between labels and integer indices
    # This simplifies SAT variable indexing
    label_to_idx = Dict(label => idx for (idx, label) in enumerate(unique_labels))
    idx_to_label = Dict(idx => label for (idx, label) in enumerate(unique_labels))

    # Try increasing depths until solution found
    for depth = 1:max_depth_limit
        println("Trying depth: $depth")

        # Create complete binary tree of current depth
        t = Dict{Int,NodeT}()
        completeTree(t, depth)
        leaves = [node for (id, node) in t if node.leaf]

        # Compute ancestor relationships for path encoding
        Al = computeLeftAncestors(t, leaves)
        Ar = computeRightAncestors(t, leaves)

        # Collect node IDs for variable creation
        internal_node_ids = [node.t for (node_id, node) in t if node.leaf == false]
        leaf_ids = [leaf.t for leaf in leaves]

        # =====================================================================
        # SAT VARIABLE CREATION
        # =====================================================================

        # a[t,j]: Does internal node t use feature j?
        # β(t) = j iff a[t,j] = true
        @satvariable(a[internal_node_ids, 1:max_c], Bool)

        # s[i,t]: Does sample i go LEFT at internal node t?
        # If false, sample goes RIGHT
        @satvariable(s[1:max_r, internal_node_ids], Bool)

        # z[i,l]: Does sample i reach leaf l?
        # Exactly one should be true for each sample
        @satvariable(z[1:max_r, leaf_ids], Bool)

        # g[l,c]: Does leaf l predict class c?
        # θ(l) = c iff g[l,c] = true
        @satvariable(g[leaf_ids, 1:n_labels], Bool)

        # Collect all clauses to be solved
        clauses = BoolExpr[]

        # =====================================================================
        # CLAUSES FOR INTERNAL NODES
        # =====================================================================

        for node_t in internal_node_ids
            # -----------------------------------------------------------------
            # CLAUSE 1 & 2: Exactly one feature per node
            # -----------------------------------------------------------------
            # Each internal node must select exactly one feature for splitting
            # This encodes the β function: β: InternalNodes → Features

            feature_vars = [a[node_t, j] for j in 1:max_c]

            # At least one feature must be selected
            push!(clauses, reduce(∨, feature_vars))

            # At most one feature (pairwise mutual exclusion)
            for i = 1:length(feature_vars)
                for j = (i+1):length(feature_vars)
                    push!(clauses, ¬feature_vars[i] ∨ ¬feature_vars[j])
                end
            end

            # -----------------------------------------------------------------
            # CLAUSES 3, 4, 9, 10: Feature-specific constraints
            # -----------------------------------------------------------------
            # For each possible feature, encode ordering and boundary constraints

            for j in 1:max_c
                # Sort sample indices by feature j values
                # This allows us to enforce monotonicity: if x[i,j] < x[k,j],
                # and both go left, then there's a consistent ordering
                sorted_indices = sortperm(features_train[:, j])
                ba = a[node_t, j]  # Is this feature selected?

                # -------------------------------------------------------------
                # CLAUSE 3: Ordering consistency for consecutive samples
                # -------------------------------------------------------------
                # If feature j is selected and values differ, maintain order:
                # x[prev,j] < x[curr,j] AND curr goes LEFT => prev goes LEFT
                # Logically: a[t,j] ∧ s[curr,t] ∧ ¬s[prev,t] => FALSE
                # CNF form: ¬a[t,j] ∨ ¬s[curr,t] ∨ s[prev,t]

                for i = 2:length(sorted_indices)
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    bs_curr = s[curr_idx, node_t]
                    bs_prev = s[prev_idx, node_t]

                    if features_train[prev_idx, j] != features_train[curr_idx, j]
                        # Different values: enforce ordering
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    else
                        # -----------------------------------------------------
                        # CLAUSE 4: Equal values go same direction
                        # -----------------------------------------------------
                        # If values are equal, they must be treated identically
                        # Both go same direction (either both left or both right)
                        push!(clauses, ¬ba ∨ ¬bs_prev ∨ bs_curr)
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    end
                end

                # -------------------------------------------------------------
                # CLAUSE 9: Minimum value goes LEFT
                # -------------------------------------------------------------
                # The smallest value for feature j must go to the left child
                # This establishes a lower bound for the split
                if !isempty(sorted_indices)
                    first_idx = sorted_indices[1]
                    bs_first = s[first_idx, node_t]
                    push!(clauses, ¬ba ∨ bs_first)

                    # ---------------------------------------------------------
                    # CLAUSE 10: Maximum value goes RIGHT
                    # ---------------------------------------------------------
                    # The largest value for feature j must go to the right child
                    # This establishes an upper bound for the split
                    last_idx = sorted_indices[end]
                    bs_last = s[last_idx, node_t]
                    push!(clauses, ¬ba ∨ ¬bs_last)
                end
            end
        end

        # =====================================================================
        # CLAUSES FOR LEAF NODES
        # =====================================================================

        for leaf_t in leaf_ids
            # Convert absolute leaf id to relative leaf index (1-based)
            # Leaves at depth d are numbered 2^d to 2^(d+1)-1
            # We need indices 1 to 2^d for our variables
            ll = leaf_t - 2^(depth) + 1

            # -----------------------------------------------------------------
            # CLAUSE 5: Left path constraint
            # -----------------------------------------------------------------
            # If sample i reaches leaf l, it must have gone LEFT at all
            # ancestors where l is in the left subtree
            # Logically: z[i,l] => s[i,p] for all p in Al[l]
            # CNF form: ¬z[i,l] ∨ s[i,p]

            for p in Al[leaf_t]
                for ind = 1:max_r
                    bs = s[ind, p]
                    bz = z[ind, ll]
                    push!(clauses, ¬bz ∨ bs)
                end
            end

            # -----------------------------------------------------------------
            # CLAUSE 6: Right path constraint
            # -----------------------------------------------------------------
            # If sample i reaches leaf l, it must have gone RIGHT at all
            # ancestors where l is in the right subtree
            # Logically: z[i,l] => ¬s[i,p] for all p in Ar[l]
            # CNF form: ¬z[i,l] ∨ ¬s[i,p]

            for p in Ar[leaf_t]
                for ind = 1:max_r
                    bs = s[ind, p]
                    bz = z[ind, ll]
                    push!(clauses, ¬bz ∨ ¬bs)
                end
            end

            # -----------------------------------------------------------------
            # CLAUSE 7: Complete path definition
            # -----------------------------------------------------------------
            # Ensures that if a sample follows the correct path (left at all
            # left ancestors, right at all right ancestors), then it MUST
            # reach this leaf. This is the converse of clauses 5 and 6.
            # Logically: (∀p∈Al: s[i,p]) ∧ (∀p∈Ar: ¬s[i,p]) => z[i,l]
            # CNF form: z[i,l] ∨ (∃p∈Al: ¬s[i,p]) ∨ (∃p∈Ar: s[i,p])

            for ind = 1:max_r
                left_ancestors = Al[leaf_t]
                right_ancestors = Ar[leaf_t]
                bz = z[ind, ll]

                # Collect violations: any deviation from correct path
                violations = BoolExpr[]
                for p in left_ancestors
                    push!(violations, ¬s[ind, p])  # Should go left but doesn't
                end
                for p in right_ancestors
                    push!(violations, s[ind, p])   # Should go right but doesn't
                end

                if !isempty(violations)
                    push!(clauses, bz ∨ reduce(∨, violations))
                end
            end

            # -----------------------------------------------------------------
            # CLAUSE 8: At most one label per leaf
            # -----------------------------------------------------------------
            # Each leaf can predict at most one class
            # (Note: "at most" allows for no label, which will be UNSAT)
            # This encodes: θ: Leaves → Classes is a function

            label_vars = [g[ll, c_idx] for c_idx in 1:n_labels]
            for i = 1:length(label_vars)
                for j = (i+1):length(label_vars)
                    push!(clauses, ¬label_vars[i] ∨ ¬label_vars[j])
                end
            end

            # -----------------------------------------------------------------
            # CLAUSE 11: Correct classification (Min-Depth)
            # -----------------------------------------------------------------
            # If sample i reaches leaf l, then leaf l MUST predict the
            # correct class for sample i. This is the key constraint that
            # enforces 100% training accuracy.
            # Logically: z[i,l] => g[l, correct_class(i)]
            # CNF form: ¬z[i,l] ∨ g[l, correct_class(i)]

            for ind = 1:max_r
                correct_label = labels_train[ind]
                correct_label_idx = label_to_idx[correct_label]
                bz = z[ind, ll]
                bg = g[ll, correct_label_idx]
                push!(clauses, ¬bz ∨ bg)
            end
        end

        # =====================================================================
        # GLOBAL CONSTRAINT: Each sample reaches exactly one leaf
        # =====================================================================
        # This ensures the decision tree is a proper partition of the space

        for ind = 1:max_r
            leaf_vars = [z[ind, a] for a in 1:length(leaf_ids)]

            # At least one leaf (every sample must end somewhere)
            push!(clauses, reduce(∨, leaf_vars))

            # At most one leaf (samples can't reach multiple leaves)
            for i = 1:length(leaf_vars)
                for j = (i+1):length(leaf_vars)
                    push!(clauses, ¬leaf_vars[i] ∨ ¬leaf_vars[j])
                end
            end
        end

        # =====================================================================
        # SOLVE SAT
        # =====================================================================

        println("Solving SAT with $(length(clauses)) clauses...")
        clauses_expr = reduce(∧, clauses)

        try
            status = sat!(clauses_expr, solver = Z3())

            if status == :SAT
                println("Solution found at depth $depth!")
                return decodeSolution(t, a, s, z, g, features_train, labels_train,
                                    Al, Ar, label_to_idx, idx_to_label)
            else
                println("No solution at depth $depth (UNSAT)")
            end
        catch e
            println("Error during SAT solving: $e")
            return nothing
        end
    end

    println("No solution found up to depth $max_depth_limit")
    return nothing
end


#=============================================================================
    Max-Accuracy Optimal Decision Tree (TODO)
=============================================================================#

"""
    solveMaxAccuracyOptimalDT(features_train, labels_train, target_depth)

Find the maximum-accuracy decision tree for a fixed depth.

# TODO: Implementation Notes
This variant uses MaxSAT (Maximum Satisfiability) instead of standard SAT.
The key differences from Min-Depth:

1. **Fixed Depth**: The tree depth is predetermined, not minimized
2. **Soft Constraints**: Correct classifications become "soft" constraints
3. **Optimization**: Maximize number of satisfied soft constraints

# Approach
- **Hard Clauses**: All structural constraints (1-10) remain mandatory
- **Soft Clauses**: Replace Clause 11 with:
  - Create variable p[i] for each sample i
  - Hard: p[i] => (if i reaches leaf l, then l predicts correct class)
  - Soft: p[i] (maximize number of correctly classified samples)

# MaxSAT Formulation
```
Hard constraints: Φ_hard (all clauses 1-10)
Soft constraints: {p[1], p[2], ..., p[n]}

Maximize: |{i : p[i] is true}|
Subject to: Φ_hard is satisfied
```

# Solver Requirements
Requires a MaxSAT solver such as:
- RC2 (uses SAT solver iteratively)
- Open-WBO
- MaxHS

# Example Usage (when implemented)
```julia
# Find best tree of depth 3 for noisy data
solution = solveMaxAccuracyOptimalDT(features, labels, 3)
println("Accuracy: ", solution.accuracy)  # May be < 1.0
```

# Arguments
- `features_train`: Training feature matrix
- `labels_train`: Training label vector
- `target_depth::Int`: Fixed tree depth to use

# Returns
Same format as solveMinDepthOptimalDT, but accuracy may be < 1.0
"""
function solveMaxAccuracyOptimalDT(features_train, labels_train, target_depth)
    max_r, max_c = size(features_train)
    unique_labels = unique(labels_train)

    # Create tree structure
    t = Dict{Int,NodeT}()
    completeTree(t, target_depth)
    leaves = collect(Leaves(t[1]))

    # Compute ancestors
    Al = computeLeftAncestors(t, leaves)
    Ar = computeRightAncestors(t, leaves)

    # Create SAT variables (same as Min-Depth)
    a = Dict{Tuple{Int,Int},AbstractExpr}()
    s = Dict{Tuple{Int,Int},AbstractExpr}()
    z = Dict{Tuple{Int,Int},AbstractExpr}()
    g = Dict{Tuple{Int,String},AbstractExpr}()
    p = Dict{Int,AbstractExpr}()  # NEW: Variables for correct classification

    # TODO: Initialize variables
    # TODO: Add hard clauses (1-10)
    # TODO: Add soft constraint definition (Clause 12)
    # TODO: Solve with MaxSAT solver
    # TODO: Decode solution

    println("MaxSAT solver not yet implemented")
    return nothing, nothing, nothing
end


#=============================================================================
    Solution Decoding
=============================================================================#

"""
    decodeSolution(t, a, s, z, g, features_train, labels_train, Al, Ar,
                   label_to_idx, idx_to_label)

Decode SAT solution into interpretable decision tree parameters.

# Arguments
- `t`: Tree structure dictionary
- `a`: SAT variables for feature selection (β)
- `s`: SAT variables for sample routing
- `z`: SAT variables for leaf assignment
- `g`: SAT variables for leaf labels (θ)
- `features_train`: Training features
- `labels_train`: Training labels
- `Al, Ar`: Ancestor dictionaries
- `label_to_idx, idx_to_label`: Label mapping dictionaries

# Returns
Named tuple `(beta, alpha, theta, accuracy)` where:
- `beta::Dict{Int,Int}`: Maps node id → feature index
  - Example: beta[1] = 2 means root node splits on feature 2
- `alpha::Dict{Int,Float64}`: Maps node id → threshold value
  - Example: alpha[1] = 3.5 means root splits at value 3.5
  - Threshold chosen as midpoint between last LEFT and first RIGHT sample
- `theta::Dict{Int,String}`: Maps leaf id → class label
  - Example: theta[4] = "A" means leaf 4 predicts class A
- `accuracy::Float64`: Training accuracy (0.0 to 1.0)

# Decoding Process

## Step 1: Extract β (Feature Selection)
For each internal node t, find j such that a[t,j] = true
This tells us which feature node t uses for splitting.

## Step 2: Extract α (Thresholds)
For node t using feature j:
1. Sort all samples by feature j
2. Find transition point where s[i,t] changes from true to false
3. Set threshold as midpoint between last true and first false sample
4. Fallback: if no clear transition, use maximum LEFT value

## Step 3: Extract θ (Leaf Labels)
For each leaf l, find class c such that g[l,c] = true
This tells us which class leaf l predicts.

## Step 4: Verify Accuracy
For each sample i:
1. Find leaf l such that z[i,l] = true
2. Check if theta[l] matches actual label
3. Count correct predictions

# Example Output
```
=== DECODE SOLUTION ===
Selected features for each node:
  Node 1 -> Feature 1
  Node 2 -> Feature 3
  Node 3 -> Feature 2

Computed thresholds:
  Node 1 -> Threshold 5.0
  Node 2 -> Threshold 2.5
  Node 3 -> Threshold 7.2

Leaf labels:
  Leaf 4 -> Label A
  Leaf 5 -> Label B
  Leaf 6 -> Label A
  Leaf 7 -> Label B

Sample classifications:
  Sample 1: predicted=A, actual=A, correct=true
  Sample 2: predicted=B, actual=B, correct=true
  ...

Accuracy: 32/32 = 100.00%
```

# Notes
- Threshold computation assumes sorted samples have consistent routing
- For Min-Depth, accuracy should always be 100%
- For Max-Accuracy, some samples may be misclassified
"""
function decodeSolution(t, a, s, z, g, features_train, labels_train, Al, Ar,
                       label_to_idx, idx_to_label)
    println("=== DECODE SOLUTION ===")

    # Collect node IDs
    internal_node_ids = [node.t for (node_id, node) in t if node.leaf == false]
    leaf_ids = [node.t for (node_id, node) in t if node.leaf]
    max_r = size(features_train, 1)
    max_c = size(features_train, 2)
    n_labels = length(idx_to_label)

    # =========================================================================
    # DECODE β: Feature Selection
    # =========================================================================
    beta = Dict{Int,Int}()
    println("Selected features for each node:")
    for node_t in internal_node_ids
        for j in 1:max_c
            if value(a[node_t, j]) == true
                beta[node_t] = j
                println("  Node $node_t -> Feature $j")
                break
            end
        end
    end

    # =========================================================================
    # DECODE α: Thresholds
    # =========================================================================
    alpha = Dict{Int,Float64}()
    println("\nComputed thresholds:")
    for node_t in internal_node_ids
        if haskey(beta, node_t)
            feature_j = beta[node_t]
            sorted_indices = sortperm(features_train[:, feature_j])

            # Find the split point: where s[i,t] transitions from true to false
            # true = LEFT, false = RIGHT
            split_found = false
            for i = 2:length(sorted_indices)
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]

                # Check if we found the boundary: prev goes LEFT, curr goes RIGHT
                if value(s[prev_idx, node_t]) == true &&
                   value(s[curr_idx, node_t]) == false
                    # Threshold is midpoint between last LEFT and first RIGHT
                    alpha[node_t] = (features_train[prev_idx, feature_j] +
                                    features_train[curr_idx, feature_j]) / 2
                    println("  Node $node_t -> Threshold $(alpha[node_t])")
                    split_found = true
                    break
                end
            end

            # Fallback: if no clear transition found, use max LEFT value
            if !split_found
                max_left_value = -Inf
                for i in sorted_indices
                    if value(s[i, node_t]) == true
                        max_left_value = max(max_left_value, features_train[i, feature_j])
                    end
                end
                if max_left_value != -Inf
                    alpha[node_t] = max_left_value
                    println("  Node $node_t -> Threshold $(alpha[node_t]) (fallback)")
                end
            end
        end
    end

    # =========================================================================
    # DECODE θ: Leaf Labels
    # =========================================================================
    theta = Dict{Int,String}()
    println("\nLeaf labels:")
    for leaf_t in leaf_ids
        # Convert absolute leaf id to relative index
        ll = leaf_t - length(leaf_ids) + 1

        for c_idx in 1:n_labels
            if value(g[ll, c_idx]) == true
                label_str = idx_to_label[c_idx]  # Convert index back to label
                theta[leaf_t] = label_str
                println("  Leaf $ll (id=$leaf_t) -> Label $label_str")
                break
            end
        end
    end

    # =========================================================================
    # VERIFY ACCURACY
    # =========================================================================
    correct = 0
    println("\nSample classifications:")
    for ind = 1:max_r
        for leaf_t in leaf_ids
            ll = leaf_t - length(leaf_ids) + 1

            # Check if this sample reaches this leaf
            if value(z[ind, ll]) == true
                predicted = get(theta, leaf_t, "UNKNOWN")
                actual = labels_train[ind]
                is_correct = predicted == actual
                if is_correct
                    correct += 1
                end
                println("  Sample $ind: predicted=$predicted, actual=$actual, correct=$is_correct")
                break
            end
        end
    end

    accuracy = correct / length(labels_train)
    println("\nAccuracy: $correct/$(length(labels_train)) = $(round(accuracy*100, digits=2))%")

    return (beta = beta, alpha = alpha, theta = theta, accuracy = accuracy)
end


#=============================================================================
    Tree Representation Structures
=============================================================================#

"""
    TreeSAT

Represents a decision tree node in a format compatible with visualization.

# Fields
- `t::Integer`: Node identifier (BFS numbering)
- `leaf::Bool`: Whether this is a leaf node
- `feature::Union{Int,Nothing}`: Index of feature used for splitting (nothing for leaves)
- `threshold::Union{Float64,Nothing}`: Threshold value for split (nothing for leaves)
- `label::Union{String,Nothing}`: Class prediction (nothing for internal nodes)
- `children::Vector{TreeSAT}`: Child nodes (empty for leaves)

# Example
```julia
# Internal node splitting on feature 2 with threshold 3.5
internal = TreeSAT(1, false, 2, 3.5, nothing, [left_child, right_child])

# Leaf node predicting class "A"
leaf = TreeSAT(4, true, nothing, nothing, "A", [])
```
"""
struct TreeSAT
    t::Integer
    leaf::Bool
    feature::Union{Int,Nothing}
    threshold::Union{Float64,Nothing}
    label::Union{String,Nothing}
    children::Vector{TreeSAT}
end

# Enable tree traversal
AbstractTrees.children(node::TreeSAT) = node.children


"""
    printTreeSAT(alpha::Dict{Int,Float64}, beta::Dict{Int,Int},
                 theta::Dict{Int,String})

Convert SAT solution (β, α, θ) into TreeSAT structure for visualization.

# Arguments
- `alpha`: Threshold dictionary (α: nodes → thresholds)
- `beta`: Feature dictionary (β: nodes → features)
- `theta`: Label dictionary (θ: leaves → classes)

# Returns
- `Dict{Int,TreeSAT}`: Dictionary mapping node id to TreeSAT structure

# Algorithm
1. Find maximum node id to determine tree depth
2. Create leaf nodes first (bottom-up)
3. Create internal nodes, linking to children
4. Return root node (id=1)

# Example
```julia
beta = Dict(1 => 2, 2 => 1, 3 => 3)
alpha = Dict(1 => 5.0, 2 => 2.5, 3 => 7.0)
theta = Dict(4 => "A", 5 => "B", 6 => "A", 7 => "B")

tree_dict = printTreeSAT(alpha, beta, theta)
root = tree_dict[1]  # Root node of the tree
```

# Notes
- Uses BFS numbering convention
- Creates complete binary tree structure
- Compatible with AbstractTrees for traversal
"""
function printTreeSAT(alpha::Dict{Int,Float64}, beta::Dict{Int,Int},
                     theta::Dict{Int,String})
    # Find maximum node id across all dictionaries
    max_node = maximum(union(keys(alpha), union(keys(beta), keys(theta))))

    # Dictionary to store TreeSAT nodes
    n = Dict{Int,TreeSAT}()

    # Calculate tree depth from maximum node id
    max_depth = floor(Int, log2(max_node + 1)) - 1

    # Create leaf nodes first (deepest level)
    for i = (2^max_depth):(2^(max_depth+1)-1)
        n[i] = TreeSAT(i, true, nothing, nothing, theta[i], [])
    end

    # Create internal nodes from bottom to top
    for i = (2^max_depth-1):-1:1
        n[i] = TreeSAT(i, false, beta[i], alpha[i], nothing, [n[2*i], n[2*i+1]])
    end

    return n
end


#=============================================================================
    Utility Functions
=============================================================================#

"""
    majority(val)

Find the most frequent element in a collection (mode).

# Arguments
- `val`: Any collection of elements

# Returns
The most frequently occurring element. If there's a tie, returns one of them.

# Example
```julia
majority([1, 2, 2, 3, 2, 1])  # Returns 2
majority(["A", "B", "A", "A", "C"])  # Returns "A"
```

# Used For
- Determining leaf predictions when multiple samples reach a leaf
- Handling ties in classification decisions
- Aggregating predictions in ensemble methods
"""
function majority(val)
    counts = countmap(val)  # using StatsBase
    max_count = maximum(collect(values(counts)))
    for (k, v) in counts
        if v == max_count
            return k
        end
    end
end


"""
    albero_PROVA(alpha, beta, theta)

Convert SAT solution into DecisionTree.jl format for compatibility.

# Arguments
- `alpha::Dict{Int,Float64}`: Thresholds for internal nodes
- `beta::Dict{Int,Int}`: Feature indices for internal nodes
- `theta::Dict{Int,String}`: Labels for leaf nodes

# Returns
- `Union{Node, Leaf}`: Root node in DecisionTree.jl format

# Description
This function bridges the gap between the SAT-based representation and
the standard DecisionTree.jl library format, allowing:
- Use of DecisionTree.jl's print_tree function
- Compatibility with SoleDecisionTreeInterface
- Integration with existing Julia ML ecosystem

# DecisionTree.jl Format
- `Node(feature, threshold, left, right)`: Internal node
- `Leaf(label, samples)`: Leaf node

# Example
```julia
beta = Dict(1 => 2, 2 => 1)
alpha = Dict(1 => 5.0, 2 => 2.5)
theta = Dict(3 => "A", 4 => "B", 5 => "C", 6 => "A")

root = albero_PROVA(alpha, beta, theta)
print_tree(root)
# Output:
# Feature 2 < 5.0 ?
# ├─ Feature 1 < 2.5 ?
# │  ├─ A
# │  └─ B
# └─ ...
```

# Handling Missing Children
If a leaf is referenced but doesn't exist in theta (incomplete tree),
creates a placeholder Leaf with "UNKNOWN" label.

# Notes
- Uses majority() for leaf labels (handles multiple samples per leaf)
- Sorts thresholds by node id for bottom-up construction
- Handles incomplete trees gracefully with UNKNOWN placeholders
"""
function albero_PROVA(alpha, beta, theta)
    # Sort by node id (descending) for bottom-up construction
    threshold = sort(collect(alpha), by=x->x[1], rev=true)
    labels = sort(collect(theta), by=x->x[1], rev=true)

    # Calculate tree depth
    max_node_id = maximum(union(keys(alpha), union(keys(beta), keys(theta))))
    max_depth = floor(Int, log2(max_node_id + 1)) - 1

    # Dictionary to store Node/Leaf objects
    n = Dict{Int, Union{Node, Leaf}}()

    # Create leaf nodes first
    for (key, values) in labels
        maj = majority(values)
        n[key] = Leaf("$maj", [values])
    end

    # Create internal nodes from bottom to top
    for (key, threshold_value) in threshold
        if haskey(n, 2*key) && haskey(n, 2*key+1)
            # Both children exist
            n[key] = Node(beta[key], threshold_value, n[2*key], n[2*key+1])
        elseif haskey(n, 2*key)
            # Only left child exists
            n[key] = Node(beta[key], threshold_value, n[2*key],
                         Leaf("UNKNOWN", ["UNKNOWN"]))
        elseif haskey(n, 2*key+1)
            # Only right child exists
            n[key] = Node(beta[key], threshold_value,
                         Leaf("UNKNOWN", ["UNKNOWN"]), n[2*key+1])
        end
    end

    root = n[1]
    return root
end


#=============================================================================
    Example Dataset
=============================================================================#

"""
    create_simple_dummy_dataset_depth5()

Create a synthetic dataset requiring depth-5 tree for perfect classification.

# Returns
- `features_matrix`: 32×5 matrix of binary features
- `labels`: Vector of 32 labels ("A", "B", or "C")

# Dataset Properties
- **Size**: 32 samples, 5 binary features
- **Classes**: 3 classes (A, B, C)
- **Minimum Depth**: Requires depth 5 for perfect separation
- **Pattern**: Each combination of feature values maps to a specific class
- **Complexity**: Designed to test SAT solver on non-trivial problem

# Feature Space
All possible combinations of 5 binary features (2^5 = 32 samples):
- Feature 1: {0, 1}
- Feature 2: {0, 1}
- Feature 3: {0, 1}
- Feature 4: {0, 1}
- Feature 5: {0, 1}

# Label Assignment
Labels are assigned in a pattern that requires using all 5 features
to achieve perfect classification. The pattern is designed such that:
- Simple depth-2 or depth-3 trees cannot separate all classes
- Optimal tree needs to consider complex feature interactions
- Tests the SAT solver's ability to find deep optimal trees

# Example Usage
```julia
features, labels = create_simple_dummy_dataset_depth5()
println("Dataset size: ", size(features))  # (32, 5)
println("Unique classes: ", unique(labels))  # ["A", "B", "C"]

# Solve for optimal tree
solution = solveMinDepthOptimalDT(features, labels, 10)
```

# Visualization of Sample Data
```
Sample | f1 f2 f3 f4 f5 | Label
-------|----------------|-------
   1   | 0  0  0  0  0  |   A
   2   | 0  0  0  0  1  |   B
   3   | 0  0  0  1  0  |   A
   4   | 0  0  0  1  1  |   C
  ...  |     ...        |  ...
  32   | 1  1  1  1  1  |   A
```

# Notes
- All features are binary (0 or 1)
- No missing values
- Balanced dataset (classes appear ~equally often)
- Deterministic (always produces same data)
"""
function create_simple_dummy_dataset_depth5()
    # Binary feature combinations
    features = [
        # f1 f2 f3 f4 f5 -> Label
        [0, 0, 0, 0, 0],  # A
        [0, 0, 0, 0, 1],  # B
        [0, 0, 0, 1, 0],  # A
        [0, 0, 0, 1, 1],  # C
        [0, 0, 1, 0, 0],  # B
        [0, 0, 1, 0, 1],  # A
        [0, 0, 1, 1, 0],  # C
        [0, 0, 1, 1, 1],  # B
        [0, 1, 0, 0, 0],  # C
        [0, 1, 0, 0, 1],  # A
        [0, 1, 0, 1, 0],  # B
        [0, 1, 0, 1, 1],  # C
        [0, 1, 1, 0, 0],  # A
        [0, 1, 1, 0, 1],  # B
        [0, 1, 1, 1, 0],  # A
        [0, 1, 1, 1, 1],  # C
        [1, 0, 0, 0, 0],  # B
        [1, 0, 0, 0, 1],  # C
        [1, 0, 0, 1, 0],  # A
        [1, 0, 0, 1, 1],  # B
        [1, 0, 1, 0, 0],  # C
        [1, 0, 1, 0, 1],  # B
        [1, 0, 1, 1, 0],  # A
        [1, 0, 1, 1, 1],  # C
        [1, 1, 0, 0, 0],  # A
        [1, 1, 0, 0, 1],  # B
        [1, 1, 0, 1, 0],  # C
        [1, 1, 0, 1, 1],  # A
        [1, 1, 1, 0, 0],  # B
        [1, 1, 1, 0, 1],  # C
        [1, 1, 1, 1, 0],  # B
        [1, 1, 1, 1, 1],  # A
    ]

    labels = [
        "A", "B", "A", "C", "B", "A", "C", "B",
        "C", "A", "B", "C", "A", "B", "A", "C",
        "B", "C", "A", "B", "C", "B", "A", "C",
        "A", "B", "C", "A", "B", "C", "B", "A",
    ]

    # Convert to matrix format
    features_matrix = zeros(Int, length(features), 5)
    for i = 1:length(features)
        features_matrix[i, :] = features[i]
    end

    return features_matrix, labels
end


#=============================================================================
    Main Entry Point
=============================================================================#

"""
    main()

Demonstration of SAT-based optimal decision tree learning.

# Workflow
1. Load or create training dataset
2. Solve for minimum-depth optimal tree
3. Decode SAT solution to tree parameters (β, α, θ)
4. Convert to DecisionTree.jl format
5. Display tree in multiple formats

# Output Examples

## Console Output
```
Dataset loaded: (32, 5) features, 3 classes
Trying depth: 1
Solving SAT with 245 clauses...
No solution at depth 1 (UNSAT)
Trying depth: 2
Solving SAT with 512 clauses...
No solution at depth 2 (UNSAT)
...
Trying depth: 5
Solving SAT with 3847 clauses...
Solution found at depth 5!

=== DECODE SOLUTION ===
Selected features for each node:
  Node 1 -> Feature 1
  Node 2 -> Feature 3
  ...

Computed thresholds:
  Node 1 -> Threshold 0.5
  Node 2 -> Threshold 0.5
  ...

Leaf labels:
  Leaf 1 (id=16) -> Label A
  Leaf 2 (id=17) -> Label B
  ...

Accuracy: 32/32 = 100.00%
```

## Tree Visualization (DecisionTree.jl)
```
Feature 1 < 0.5 ?
├─ Feature 3 < 0.5 ?
│  ├─ Feature 4 < 0.5 ?
│  │  ├─ A
│  │  └─ B
│  └─ ...
└─ Feature 2 < 0.5 ?
   ├─ ...
   └─ C
```

## Tree Visualization (SoleDecisionTreeInterface)
```
DecisionTree with 31 nodes
  [1] Feature1 ≤ 0.5
    [2] Feature3 ≤ 0.5
      [4] Feature4 ≤ 0.5
        [8] → A
        [9] → B
      [5] ...
    [3] Feature2 ≤ 0.5
      ...
```

# Customization
To use your own dataset:
```julia
# Replace the dataset loading line with:
features_train, labels_train = load_your_dataset()

# Adjust max depth if needed:
solution = solveMinDepthOptimalDT(features_train, labels_train, 15)
```

# Performance Tips
- Start with small datasets (< 50 samples)
- Limit max_depth_limit to 8-10 for initial experiments
- Use binary or low-cardinality features when possible
- Monitor clause count (> 10K clauses may be slow)

# Alternative Example (Commented)
The code includes a second example showing manual tree construction:
```julia
alpha = Dict(1 => 5.0, 2 => 2.5, 3 => 1.5)
beta  = Dict(1 => 1, 2 => 2, 3 => 1)
theta = Dict(4 => "L", 5 => "R", 6 => "L", 7 => "R")
root = albero_PROVA(alpha, beta, theta)
print_tree(root)
```
"""
function main()
    println("="^70)
    println("SAT-BASED OPTIMAL DECISION TREE LEARNING")
    println("="^70)

    # =========================================================================
    # EXAMPLE 1: Solve for Optimal Tree
    # =========================================================================

    # Load training data
    features_train, labels_train = create_simple_dummy_dataset_depth5()

    println("\nDataset loaded:")
    println("  Size: $(size(features_train)) features")
    println("  Classes: $(length(unique(labels_train))) unique labels")
    println("  Labels: $(unique(labels_train))")

    # Solve Min-Depth problem
    println("\n" * "="^70)
    println("SOLVING MIN-DEPTH PROBLEM")
    println("="^70)
    solution = solveMinDepthOptimalDT(features_train, labels_train, 10)

    if solution !== nothing
        beta = solution.beta
        alpha = solution.alpha
        theta = solution.theta

        println("\n" * "="^70)
        println("TREE VISUALIZATION")
        println("="^70)

        # Convert to DecisionTree.jl format
        root = albero_PROVA(alpha, beta, theta)

        println("\n--- DecisionTree.jl Format ---")
        print_tree(root)

        println("\n--- SoleDecisionTreeInterface Format ---")
        sole_tree = solemodel(root)
        printmodel(sole_tree)

        println("\n" * "="^70)
        println("SUMMARY")
        println("="^70)
        println("Final Accuracy: $(round(solution.accuracy * 100, digits=2))%")
        println("Tree Depth: $(maximum(floor(Int, log2(maximum(keys(alpha)) + 1)) - 1))")
        println("Number of Internal Nodes: $(length(beta))")
        println("Number of Leaves: $(length(theta))")
    else
        println("\n⚠ No solution found within depth limit!")
        println("Consider:")
        println("  - Increasing max_depth_limit")
        println("  - Checking for data inconsistencies")
        println("  - Using Max-Accuracy mode instead")
    end

    # =========================================================================
    # EXAMPLE 2: Manual Tree Construction (for testing)
    # =========================================================================

    println("\n\n" * "="^70)
    println("EXAMPLE: MANUAL TREE CONSTRUCTION")
    println("="^70)
    println("(Uncomment to run)\n")

    #=
    # Define tree manually
    alpha = Dict(1 => 5.0, 2 => 2.5, 3 => 1.5)
    beta  = Dict(1 => 1, 2 => 2, 3 => 1)
    theta = Dict(4 => "L", 5 => "R", 6 => "L", 7 => "R")

    println("Manual tree parameters:")
    println("  Beta (features): $beta")
    println("  Alpha (thresholds): $alpha")
    println("  Theta (labels): $theta")

    root = albero_PROVA(alpha, beta, theta)

    println("\n--- DecisionTree.jl Format ---")
    print_tree(root)

    println("\n--- SoleDecisionTreeInterface Format ---")
    sole_tree = solemodel(root)
    printmodel(sole_tree)
    =#
end

# Run the demonstration
main()
