"""
    part_nmf(X, A1, l; init=:nndsvdar, alg=:mult, maxiter=50000)  

Implementation of PartNMF as described in: 
> M. S. Karoui, F. z. Benhalouche, Y. Deville, K. Djerriri, X. Briottet and A. L. Bris, "Detection And Area Estimation For Photovoltaic Panels In Urban Hyperspectral Remote Sensing Data By An Original Nmf-Based Unmixing Method," IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, Valencia, Spain, 2018, pp. 1640-1643, doi: 10.1109/IGARSS.2018.8518204.
"""
function part_nmf(
    X::AbstractMatrix{T},
    A1::AbstractMatrix{T},
    l::Integer;
    init::Symbol = :nndsvdar,
    alg::Symbol = :mult,
    maxiter::Integer = 50000,
    tol::Real = cbrt(eps(T)/100),
    ε::Real = eps(T),
    A2_0::Union{AbstractMatrix{T}, Nothing} = nothing,
    S_0::Union{AbstractMatrix{T}, Nothing} = nothing,
    constantS::Bool = true
) where T
    # check for non-negativity
    eltype(X) <: Number && all(x -> x >= zero(T), X) || throw(ArgumentError(
        "X must be non-negative."
    ))
    eltype(A1) <: Number && all(x -> x >= zero(T), A1) || throw(ArgumentError(
        "A must be nonnegative"
    ))

    n, k = size(X)

    # check size of A
    n_A, l_1 = size(A1)
    n == n_A && l_1 < l || throw(ArgumentError("Invalid size for A."))

    # required size for A2_0
    l_2 = l - l_1

    if init == :custom
        !isnothing(A2_0) && !isnothing(S_0) || throw(ArgumentError(
            "For :custom initialization, set A2_0 and S_0."
        ))

        # ensure nonnegativity
        eltype(A2_0) <: Number && all(x -> x >= zero(T), A2_0) || throw(
            ArgumentError("A2_0 must be non-negative.")
        )
        eltype(S_0) <: Number && all(x -> x >= zero(T), S_0) || throw(
            ArgumentError("S_0 must be non-negative.")
        )

        # check sizes
        n0, l_2_0 = size(A2_0) 
        n == n0 && l_2 == l_2_0 || throw(ArgumentError("Invalid size for A2_0."))
        l0, k0 = size(S_0) 
        k == k0 && l == l0 || throw(ArgumentError("Invalid size for S_0."))
    else
        isnothing(A2_0) && isnothing(S_0) || @warn "A2_0 and S_0 will be ignored unless init = :custom."
    end # if

    # initalization
    if init == :rand
        A2, S = randinit(X, l_2, l)
    elseif init == :nndsvd
        A2, S = nndsvd(X, l_2, l; constantS=constantS, variant=:std)
    elseif init == :nndsvda
        A2, S = nndsvd(X, l_2, l; constantS=constantS, variant=:a)
    elseif init == :nndsvdar
        A2, S = nndsvd(X, l_2, l; constantS=constantS, variant=:ar)
    elseif init == :custom
        A2, S = A2_0, S_0
    else
        throw(ArgumentError("Unknown initialization: $init."))
    end # if
    A1 = A1::Matrix{T}
    A2 = A2::Matrix{T}
    S = S::Matrix{T}

    # expand A1 and A2 to be n×l
    A1 = hcat(A1, zeros(T, n, l_2))
    A2 = hcat(zeros(T, n, l_1), A2)

    # pick algorithm
    if alg == :mult
        ret = solve!(
            MultUpdate{T}(; maxiter=maxiter, ε=ε, tol=tol),
            X,
            A1,
            A2,
            S
        )
    else 
        throw(ArgumentError("Unknown algorithm: $alg."))
    end # if

    return ret
end # function part_nmf
