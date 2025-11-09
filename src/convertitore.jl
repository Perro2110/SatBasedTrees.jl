using SoleLogics
using Satisfiability
using SoleLogics: Atom, ¬, ∧, ∨

function replace_sat(f, atom_to_var)    
    if f isa Atom
        return atom_to_var[f]
    elseif f isa SyntaxBranch
        tok = token(f)
        ch = children(f)
        converted = [replace_sat(c, atom_to_var) for c in ch]

        if tok == ¬
            # println("$tok - not")
            return Satisfiability.not(converted[1])
        elseif tok == ∧
            # println("$tok - and")
            return Satisfiability.and(converted...)
        elseif tok == ∨
            # println("$tok - or")
            return Satisfiability.or(converted...)
        else
            # println("$tok - Non va bene")
            return f  # Se non è un tipo riconosciuto, restituisci così com'è
        end
    else
        error("Tipo di formula non gestito: $(typeof(f))")
    end
end

function convertitore(formula::Formula)
    # spezzo la formula in subformule
    subformule = subformulas(formula)
    """
    println("subformule: ")
    for s in subformule
        println(s)
    end
    """

    # converto ogni subformula in atomi e le metto in una lista
    atomi = [f for f in subformule if f isa Atom]

    # assegno satvariable a ogni atomo (unici nel dizionario)
    atom_to_var = Dict{Atom, AbstractExpr}()  # Dizionario per mappare atomi a variabili
    for a in atomi
        if !haskey(atom_to_var, a)
            atom_to_var[a] = @satvariable(b, Bool)
        end
    end
    # println("Dizionario atomi a variabili: ", atom_to_var)

    formula2 = replace_sat(formula, atom_to_var)
    return formula2, atom_to_var
end

function main()
    # Esempio di formula
    ff = ¬Atom("p") ∧ Atom("q") ∧ (¬Atom("s") ∧ ¬Atom("z") ∨ Atom("t")) # insoddisfacibile
    # ff = (Atom("p") ) ∧ (Atom("q") ∨ ¬Atom("r")) # soddisfacibile
    println("Formula: ", ff)
    
    # converte la formula in CNF per Satisfiability.jl
    form_tradotta, atomi_satvariable = convertitore(ff)
    # println("Formula convertita: ", form_tradotta)
    # println("Dizionario atomi a variabili: ", atomi_satvariable)

    # sat
    status = sat!(form_tradotta, solver=Z3())
    if status == :SAT
        println("Formula soddisfacibile")
    else
        println("Formula insoddisfacibile")
    end
end

main()