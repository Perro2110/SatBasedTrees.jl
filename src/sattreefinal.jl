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

struct NodeT
    t::Integer
    leaf::Bool
    children::Vector{NodeT}
end
AbstractTrees.children(node::NodeT) = node.children

"""
Crea un albero binario completo di profondità max_depth
I nodi sono numerati in ordine BFS: 1 è la radice, 2i e 2i+1 sono i figli di i
"""
function completeTree(t::Dict{Int,NodeT}, max_depth::Int)
    # Prima creiamo le foglie (livello più profondo)
    for i = (2^max_depth):(2^(max_depth+1)-1)
        t[i] = NodeT(i, true, [])
    end
    # Poi creiamo i nodi interni dal basso verso l'alto
    for i = (2^max_depth-1):-1:1
        t[i] = NodeT(i, false, [t[2*i], t[2*i+1]])
    end
end

"""
Calcola gli antenati sinistri per ogni foglia
Al[t] contiene tutti i nodi antenati di t tali che t è raggiungibile dal loro ramo sinistro
"""
function computeLeftAncestors(t::Dict{Int,NodeT}, leaves)
    Al = Dict{Int,Vector{Int}}()

    for leaf in leaves
        Al[leaf.t] = Int[]
        current = leaf.t

        # Risali l'albero fino alla radice
        while current > 1
            parent = div(current, 2)
            # Se current è figlio sinistro del parent
            if current == 2 * parent
                push!(Al[leaf.t], parent)
            end
            current = parent
        end
    end
    return Al
end

"""
Calcola gli antenati destri per ogni foglia
Ar[t] contiene tutti i nodi antenati di t tali che t è raggiungibile dal loro ramo destro
"""
function computeRightAncestors(t::Dict{Int,NodeT}, leaves)
    Ar = Dict{Int,Vector{Int}}()

    for leaf in leaves
        Ar[leaf.t] = Int[]
        current = leaf.t

        # Risali l'albero fino alla radice
        while current > 1
            parent = div(current, 2)
            # Se current è figlio destro del parent
            if current == 2 * parent + 1
                push!(Ar[leaf.t], parent)
            end
            current = parent
        end
    end
    return Ar
end

"""
Risolve il problema Min-Depth: trova l'albero di profondità minima che classifica
correttamente tutti i punti di training
"""
function solveMinDepthOptimalDT(features_train, labels_train, max_depth_limit = 10)
    max_r, max_c = size(features_train)
    unique_labels = unique(labels_train)
    n_labels = length(unique_labels)

    # Crea mappatura label -> indice intero
    label_to_idx = Dict(label => idx for (idx, label) in enumerate(unique_labels))
    idx_to_label = Dict(idx => label for (idx, label) in enumerate(unique_labels))

    # Prova profondità crescenti finché non trova una soluzione
    for depth = 1:max_depth_limit
        println("Provando profondità: $depth")

        # Crea l'albero
        t = Dict{Int,NodeT}()
        completeTree(t, depth)
        leaves = [NodeT for (id, NodeT) in t if NodeT.leaf]

        # Calcola antenati
        Al = computeLeftAncestors(t, leaves)
        Ar = computeRightAncestors(t, leaves)

        # Raccogli gli ID dei nodi interni e delle foglie
        internal_NodeT_ids = [NodeT.t for (NodeT_id, NodeT) in t if NodeT.leaf == false]
        leaf_ids = [leaf.t for leaf in leaves]

        # Crea variabili SAT
        @satvariable(a[internal_NodeT_ids, 1:max_c], Bool)
        @satvariable(s[1:max_r, internal_NodeT_ids], Bool)
        @satvariable(z[1:max_r, leaf_ids], Bool)
        @satvariable(g[leaf_ids, 1:n_labels], Bool)  # Usa indici interi per le label

        # Raccoglie tutte le clausole
        clauses = BoolExpr[]

        # Clausole per ogni nodo interno
        for NodeT_t in internal_NodeT_ids
            # Clausola 1 & 2: Esattamente una feature per nodo
            feature_vars = [a[NodeT_t, j] for j in 1:max_c]

            # Almeno una feature
            push!(clauses, reduce(∨, feature_vars))

            # Al massimo una feature (pairwise constraints)
            for i = 1:length(feature_vars)
                for j = (i+1):length(feature_vars)
                    push!(clauses, ¬feature_vars[i] ∨ ¬feature_vars[j])
                end
            end

            for j in 1:max_c
                # Ordina i punti per la feature j
                sorted_indices = sortperm(features_train[:, j])
                ba = a[NodeT_t, j]

                # Clausola 3: Ordinamento coerente per coppie consecutive
                for i = 2:length(sorted_indices)
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    bs_curr = s[curr_idx, NodeT_t]
                    bs_prev = s[prev_idx, NodeT_t]

                    # Se feature j è selezionata e i valori sono diversi,
                    # allora il precedente deve andare a sinistra se il corrente va a sinistra
                    if features_train[prev_idx, j] != features_train[curr_idx, j]
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    else
                        # Clausola 4: Valori uguali vanno nella stessa direzione
                        push!(clauses, ¬ba ∨ ¬bs_prev ∨ bs_curr)
                        push!(clauses, ¬ba ∨ bs_prev ∨ ¬bs_curr)
                    end
                end

                # Clausola 9: Il primo elemento va a sinistra
                if !isempty(sorted_indices)
                    first_idx = sorted_indices[1]
                    bs_first = s[first_idx, NodeT_t]
                    push!(clauses, ¬ba ∨ bs_first)

                    # Clausola 10: L'ultimo elemento va a destra
                    last_idx = sorted_indices[end]
                    bs_last = s[last_idx, NodeT_t]
                    push!(clauses, ¬ba ∨ ¬bs_last)
                end
            end
        end

        # Clausole per le foglie

        for leaf_t in leaf_ids
            #@show leaf_ids
            # Clausola 5: Percorso sinistro
            # ll = indice della prima foglia nella lista delle foglie
            ll = leaf_t - 2^(depth) + 1
            for p in Al[leaf_t]
                for ind = 1:max_r
                    bs = s[ind, p]
                    #@show ll
                    bz = z[ind, ll]
                    push!(clauses, ¬bz ∨ bs)
                end
            end

            # Clausola 6: Percorso destro
            for p in Ar[leaf_t]
                for ind = 1:max_r
                    #@show max_r
                    #@show leaf_t
                    bs = s[ind, p]
                    bz = z[ind, ll]
                    push!(clauses, ¬bz ∨ ¬bs)
                end
            end


            # Clausola 7: Definizione di percorso completo
            for ind = 1:max_r

                left_ancestors = Al[leaf_t]
                right_ancestors = Ar[leaf_t]
                bz = z[ind, ll]

                violations = BoolExpr[]
                for p in left_ancestors
                    push!(violations, ¬s[ind, p])
                end
                for p in right_ancestors
                    push!(violations, s[ind, p])
                end

                if !isempty(violations)
                    push!(clauses, bz ∨ reduce(∨, violations))
                end
            end

            # Clausola 8: Al massimo una etichetta per foglia

            label_vars = [g[ll, c_idx] for c_idx in 1:n_labels]
            for i = 1:length(label_vars)
                for j = (i+1):length(label_vars)
                    push!(clauses, ¬label_vars[i] ∨ ¬label_vars[j])
                end
            end

            # Clausola 11: Classificazione corretta (Min-Depth)
            for ind = 1:max_r
                correct_label = labels_train[ind]
                correct_label_idx = label_to_idx[correct_label]  # Converti label a indice
                bz = z[ind, ll]
                bg = g[ll, correct_label_idx]
                push!(clauses, ¬bz ∨ bg)
            end
        end

        # Ogni punto deve finire esattamente in una foglia
        for ind = 1:max_r

            leaf_vars = [z[ind, a] for a in 1:length(leaf_ids)]

            # Almeno una foglia
            push!(clauses, reduce(∨, leaf_vars))

            # Al massimo una foglia
            for i = 1:length(leaf_vars)
                for j = (i+1):length(leaf_vars)
                    push!(clauses, ¬leaf_vars[i] ∨ ¬leaf_vars[j])
                end
            end
        end

        # Risolvi SAT
        println("Risolvendo SAT con $(length(clauses)) clausole...")
        clauses_expr = reduce(∧, clauses)
        try
            status = sat!(clauses_expr, solver = Z3())

            if status == :SAT
                println("Soluzione trovata a profondità $depth !")
                return decodeSolution(t, a, s, z, g, features_train, labels_train, Al, Ar, label_to_idx, idx_to_label)
            else
                println("Nessuna soluzione a profondità $depth")
            end
        catch e
            println("Errore durante la risoluzione SAT: $e")
            return nothing
        end
    end

    println("Nessuna soluzione trovata fino alla profondità $max_depth_limit")
    return nothing
end


#TODO:
"""
Risolve il problema Max-Accuracy: trova l'albero di profondità fissa che massimizza
il numero di punti classificati correttamente
"""
function solveMaxAccuracyOptimalDT(features_train, labels_train, target_depth)
    max_r, max_c = size(features_train)
    unique_labels = unique(labels_train)

    # Crea l'albero
    t = Dict{Int,NodeT}()
    completeTree(t, target_depth)
    leaves = collect(Leaves(t[1]))

    # Calcola antenati
    Al = computeLeftAncestors(t, leaves)
    Ar = computeRightAncestors(t, leaves)

    # Crea variabili SAT
    a = Dict{Tuple{Int,Int},AbstractExpr}()
    s = Dict{Tuple{Int,Int},AbstractExpr}()
    z = Dict{Tuple{Int,Int},AbstractExpr}()
    g = Dict{Tuple{Int,String},AbstractExpr}()
    p = Dict{Int,AbstractExpr}() # Variabili per punti classificati correttamente

    # Inizializza variabili (stesso codice del Min-Depth)
    f = 1:max_c
    for j in f
        for (i, NodeT) in t
            if NodeT.leaf == false
                a[(NodeT.t, j)] = @satvariable(b, Bool)
            end
        end
    end

    for ind = 1:max_r
        for (i, NodeT) in t
            if NodeT.leaf == false
                s[(ind, NodeT.t)] = @satvariable(b, Bool)
            end
        end
    end

    for ind = 1:max_r
        for leaf in leaves
            z[(ind, leaf.t)] = @satvariable(b, Bool)
        end
        # Variabile per classificazione corretta
        p[ind] = @satvariable(b, Bool)
    end

    for c in unique_labels
        for leaf in leaves
            g[(leaf.t, c)] = @satvariable(b, Bool)
        end
    end

    # Clausole hard (stesse del Min-Depth tranne la 11)
    hard_clauses = AbstractExpr[]

    # [Inserire qui tutte le clausole 1-10 come sopra, senza la 11]
    # ... (stesso codice delle clausole 1-10 del Min-Depth)

    # Clausola 12: Definizione di classificazione corretta
    for ind = 1:max_r
        correct_label = labels_train[ind]
        bp = p[ind]
        for leaf in leaves
            bz = z[(ind, leaf.t)]
            bg = g[(leaf.t, correct_label)]
            push!(hard_clauses, (¬bp) ∨ (¬bz) ∨ (bg))
        end
    end

    # Clausole soft: Massimizza punti classificati correttamente
    soft_clauses = [p[ind] for ind = 1:max_r]

    println("Risolvendo MaxSAT...")
    # Qui dovresti usare un solver MaxSAT
    # status = maxsat!(hard_clauses, soft_clauses, solver=...)

    return nothing, nothing, nothing # Da implementare con solver MaxSAT
end
#ENDTODO


"""
Decodifica la soluzione SAT in un albero decisionale leggibile
"""
function decodeSolution(t, a, s, z, g, features_train, labels_train, Al, Ar, label_to_idx, idx_to_label)
    println("=== DECODIFICA SOLUZIONE ===")

    # Raccogli gli ID dei nodi interni e delle foglie
    internal_NodeT_ids = [NodeT.t for (NodeT_id, NodeT) in t if NodeT.leaf == false]
    leaf_ids = [NodeT.t for (NodeT_id, NodeT) in t if NodeT.leaf]
    max_r = size(features_train, 1)
    max_c = size(features_train, 2)
    n_labels = length(idx_to_label)

    # Decodifica feature selection (β)
    beta = Dict{Int,Int}()
    println("Feature selezionate per ogni nodo:")
    for NodeT_t in internal_NodeT_ids
        for j in 1:max_c
            if value(a[NodeT_t, j]) == true
                beta[NodeT_t] = j
                println("  Nodo $NodeT_t -> Feature $j")
                break
            end
        end
    end

    # Decodifica soglie (α)
    alpha = Dict{Int,Float64}()
    println("\nSoglie calcolate:")
    for NodeT_t in internal_NodeT_ids
        if haskey(beta, NodeT_t)
            feature_j = beta[NodeT_t]
            sorted_indices = sortperm(features_train[:, feature_j])

            # Trova il punto di split
            split_found = false
            for i = 2:length(sorted_indices)
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]

                if value(s[prev_idx, NodeT_t]) == true &&
                   value(s[curr_idx, NodeT_t]) == false
                    alpha[NodeT_t] = (features_train[prev_idx, feature_j] +
                                    features_train[curr_idx, feature_j]) / 2
                    println("  Nodo $NodeT_t -> Soglia $(alpha[NodeT_t])")
                    split_found = true
                    break
                end
            end

            if !split_found
                # Fallback: usa il valore massimo che va a sinistra
                max_left_value = -Inf
                for i in sorted_indices
                    if value(s[i, NodeT_t]) == true
                        max_left_value = max(max_left_value, features_train[i, feature_j])
                    end
                end
                if max_left_value != -Inf
                    alpha[NodeT_t] = max_left_value
                    println("  Nodo $NodeT_t -> Soglia $(alpha[NodeT_t]) (fallback)")
                end
            end
        end
    end

    # Decodifica etichette foglie (θ)
    theta = Dict{Int,String}()
    println("\nEtichette delle foglie:")
    for leaf_t in leaf_ids
        # ll = leaf_t - (2^(floor(Int, log2(leaf_t + 1))) - 1) + 1
        ll = leaf_t - length(leaf_ids) + 1  # funziona correttamente con alberi completi

        for c_idx in 1:n_labels
            if value(g[ll, c_idx]) == true
                label_str = idx_to_label[c_idx]  # Converti indice -> label stringa
                theta[leaf_t] = label_str
                println("  Foglia $ll -> Etichetta $label_str")
                break
            end
        end
    end

    # Verifica accuratezza
    correct = 0
    println("\nClassificazione dei punti:")
    for ind = 1:max_r
        for leaf_t in leaf_ids
            # ll = leaf_t - (2^(floor(Int, log2(leaf_t + 1))) - 1) + 1
            ll = leaf_t - length(leaf_ids) + 1

            if value(z[ind, ll]) == true
                predicted = get(theta, leaf_t, "UNKNOWN")
                actual = labels_train[ind]
                is_correct = predicted == actual
                if is_correct
                    correct += 1
                end
                println("  Punto $ind: predetto=$predicted, reale=$actual, corretto=$is_correct")
                break
            end
        end
    end

    accuracy = correct / length(labels_train)
    println("\nAccuratezza: $correct/$(length(labels_train)) = $(round(accuracy*100, digits=2))%")

    # Restituisce i risultati della decodifica
    return (beta = beta, alpha = alpha, theta = theta, accuracy = accuracy)
end

# -------------------------------------------------------------------------------------------------
# Beta(t) = j --> se a(t,j) = 1 (Feature selezionata per nodo t)
# Alpha(t) = soglia per nodo t
# Theta(t) = c --> se g(t,c) = 1 (Etichetta della foglia t)

struct TreeSAT
    t::Integer # numero nodo
    leaf::Bool # true se foglia
    feature::Union{Int,Nothing} # indice della feature utilizzata
    threshold::Union{Float64,Nothing} # soglia per la divisione
    label::Union{String,Nothing} # etichetta della foglia
    children::Vector{TreeSAT} # figli del nodo --> vuoto se leaf=true
end
AbstractTrees.children(NodeT::TreeSAT) = NodeT.children

function printTreeSAT(alpha::Dict{Int,Float64}, beta::Dict{Int,Int}, theta::Dict{Int,String})
    # numero massimo di nodi
    max_NodeT = maximum(union(keys(alpha), union(keys(beta), keys(theta))))

    # struttura TreeSAT
    n = Dict{Int,TreeSAT}()

    # da massimo nodo a profondità
    max_depth = floor(Int, log2(max_NodeT + 1)) - 1

    # Prima creiamo le foglie (livello più profondo)
    for i = (2^max_depth):(2^(max_depth+1)-1)
        n[i] = TreeSAT(i, true, nothing, nothing, theta[i], [])
        # println("Foglia $i")
    end
    # Poi creiamo i nodi interni dal basso verso l'alto
    for i = (2^max_depth-1):-1:1
        # println("Nodo $i")
        n[i] = TreeSAT(i, false, beta[i], alpha[i], nothing, [n[2*i], n[2*i+1]])
    end
    return n
end
# -------------------------------------------------------------------------------------------------

"""
Versione più semplice: dataset con meno classi ma che richiede comunque profondità 5
"""
function create_simple_dummy_dataset_depth5()
    # Dataset più piccolo e gestibile
    features = [
        # Feature: f1 f2 f3 f4 f5
        [0, 0, 0, 0, 0],  # Classe A
        [0, 0, 0, 0, 1],  # Classe B
        [0, 0, 0, 1, 0],  # Classe A
        [0, 0, 0, 1, 1],  # Classe C
        [0, 0, 1, 0, 0],  # Classe B
        [0, 0, 1, 0, 1],  # Classe A
        [0, 0, 1, 1, 0],  # Classe C
        [0, 0, 1, 1, 1],  # Classe B
        [0, 1, 0, 0, 0],  # Classe C
        [0, 1, 0, 0, 1],  # Classe A
        [0, 1, 0, 1, 0],  # Classe B
        [0, 1, 0, 1, 1],  # Classe C
        [0, 1, 1, 0, 0],  # Classe A
        [0, 1, 1, 0, 1],  # Classe B
        [0, 1, 1, 1, 0],  # Classe A
        [0, 1, 1, 1, 1],  # Classe C
        [1, 0, 0, 0, 0],  # Classe B
        [1, 0, 0, 0, 1],  # Classe C
        [1, 0, 0, 1, 0],  # Classe A
        [1, 0, 0, 1, 1],  # Classe B
        [1, 0, 1, 0, 0],  # Classe C
        [1, 0, 1, 0, 1],  # Classe B
        [1, 0, 1, 1, 0],  # Classe A
        [1, 0, 1, 1, 1],  # Classe C
        [1, 1, 0, 0, 0],  # Classe A
        [1, 1, 0, 0, 1],  # Classe B
        [1, 1, 0, 1, 0],  # Classe C
        [1, 1, 0, 1, 1],  # Classe A
        [1, 1, 1, 0, 0],  # Classe B
        [1, 1, 1, 0, 1],  # Classe C
        [1, 1, 1, 1, 0],  # Classe B
        [1, 1, 1, 1, 1],  # Classe A
    ]

    labels = [
        "A",
        "B",
        "A",
        "C",
        "B",
        "A",
        "C",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "A",
        "C",
        "B",
        "C",
        "A",
        "B",
        "C",
        "B",
        "A",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "B",
        "A",
    ]

    # Converti in matrice
    features_matrix = zeros(Int, length(features), 5)
    for i = 1:length(features)
        features_matrix[i, :] = features[i]
    end

    return features_matrix, labels
end


# majority serve per trovare l’elemento più frequente in un array o in una collezione.
# Il valore di maggioranza
function majority(val)
    counts = countmap(val) # using StatsBase
    max = maximum(collect(values(counts)))
    for (k, v) in counts
        if v == max
            return k
        end
    end
end

function albero_PROVA(alpha, beta, theta)
    # ordino per nodo
    threshold = sort(collect(alpha), by=x->x[1], rev=true)
    labels = sort(collect(theta), by=x->x[1], rev=true)
    max_depth = floor(Int, log2(maximum(union(keys(alpha), union(keys(beta), keys(theta)))) + 1)) - 1

    n = Dict{Int, Union{Node, Leaf}}()
    # Prima creiamo le foglie (livello più profondo)
    for (key, values) in labels
        maj = majority(values)
        n[key] = Leaf("$maj", [values])
    end
    # Poi creiamo i nodi interni dal basso verso l'alto
    for (key, values) in threshold
        if haskey(n, 2*key) && haskey(n, 2*key+1)
           n[key] = Node(beta[key], values, n[2*key], n[2*key+1])
        elseif haskey(n, 2*key) # solo figlio sinistro
            n[key] = Node(beta[key], values, n[2*key], Leaf("UNKNOWN", ["UNKNOWN"]))
        elseif haskey(n, 2*key+1) # solo figlio destro
            n[key] = Node(beta[key], values, Leaf("UNKNOWN", ["UNKNOWN"]), n[2*key+1])
        end
    end
    root = n[1]
    return root
end

# Esempio d'uso
function main()
    # #=
    # Carica i dati
    features_train, labels_train = create_simple_dummy_dataset_depth5()

    println(
        "Dataset caricato: $(size(features_train)) features, $(length(unique(labels_train))) classi",
    )

    # Risolvi Min-Depth
    # solution = solveMinDepthOptimalDT(features_train, labels_train, 5)
    solution = solveMinDepthOptimalDT(features_train, labels_train, 10)


    if solution !== nothing
        beta = solution[1]
        alpha  = solution[2]
        theta = solution[3]

        # prova
        root = albero_PROVA(alpha, beta, theta)

        println("\nAlbero in formato DecisionTree.jl:")
        print_tree(root)

        println("\nAlbero in formato SoleDecisionTreeInterface:")
        sole_tree = solemodel(root)
        println(sole_tree)
    end
    # =#


    #=
    alpha = Dict(1 => 5.0, 2 => 2.5, 3 => 1.5)
    beta  = Dict(1 => 1, 2 => 2, 3 => 1)
    theta = Dict(4 => "L", 5 => "R", 6 => "L", 7 => "R")
    root = albero_PROVA(alpha, beta, theta)

    print_tree(root)

    sole_tree = solemodel(root)
    printmodel(sole_tree)
    =#

end
main()
