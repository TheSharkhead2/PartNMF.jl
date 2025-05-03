using RandomizedLinAlg

function randinit(X::AbstractMatrix{T}, l_2::Integer, l::Integer) where T
    n, k = size(X)

    A2 = rand(T, n, l_2)
    S = rand(T, l, k)

    return tuple(A2, S)
end # function randinit

"""
    nndsvd(X, l_2, l; constantS=false, variant=:std)  

Adaptation of SVD initialization for NMF from [NMF.jl](https://github.com/JuliaStats/NMF.jl)

Reference
---------
    C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head
    start for nonnegative matrix factorization. Pattern Recognition, 2007.
"""
function nndsvd(
    X::AbstractMatrix{T},
    l_2::Integer, l::Integer;
    constantS::Bool=false,
    variant::Symbol=:std
) where T
    n, k = size(X)

    variant ∈ (:std, :a, :ar) || throw(ArgumentError("Invalid variant."))

    U, Σ, V = rsvd(X, l_2)

    # TODO: finish
end # function nndsvd
