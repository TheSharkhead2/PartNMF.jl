abstract type NMFAlgorithm{T} end;

function checksize(X::Matrix{T}, A1::Matrix{T}, A2::Matrix{T}, S::Matrix{T}) where T
    n, k = size(X)

    l = size(A1,  2)

    if !(size(A1, 1) == n && size(A2) == (n, l) && size(S) == (l, k))
        throw(DimensionMismatch("Dimensions of X, A1, A2, and S are inconsistent."))
    end # if

    return (n, l, k)
end # function checksize

struct Result{T}
    A2::Matrix{T}
    S::Matrix{T}
    niters::Int
    converged::Bool

    function Result{T}(A2::Matrix{T}, S::Matrix{T}, niters::Int, converged::Bool) where T
        if size(A2, 2) != size(S, 1)
            throw(DimensionMismatch("Inner dimensions of A2 and S disagree."))
        end # if
        new{T}(A2, S, niters, converged)
    end # function Result{T}
end # struct Result{T}

function solve!(
    alg::NMFAlgorithm{T},
    X::Matrix{T},
    A1::Matrix{T},
    A2::Matrix{T},
    S::Matrix{T}
) where T
    # initialization
    state = prepare_state(alg, X, A1, A2, S)
    prev_A2 = Matrix{T}(undef, size(A2))
    prev_S = Matrix{T}(undef, size(S))

    # main loop
    converged = false
    t = 0
    while !converged && t < alg.maxiter
        t += 1
        copyto!(prev_A2, A2)
        copyto!(prev_S, S)

        # step
        update!(alg, state, X, A1, A2, S)

        # check convergence
        converged, dev = check_convergence(A2, prev_A2, S, prev_S, alg.tol)
    end # while !converged && t < alg.maxiter

    return Result{T}(A2, S, t, converged)
end # function solve!

"""
    check_convergence(A2, prev_A2, S, prev_S, eps)  

Check to see if convergence has been reached. 
Note this convergence code is originally from [NMF.jl](https://github.com/JuliaStats/NMF.jl).
"""
function check_convergence(
    A2::AbstractArray{T},
    prev_A2::AbstractArray{T},
    S::AbstractArray{T},
    prev_S::AbstractArray{T},
    eps::AbstractFloat
) where T
    devmax = zero(T)

    # TODO: @inbounds
    for j in axes(A2, 2)
        dev_a2 = sum_a2 = zero(T)
        for i in axes(A2, 1)
            dev_a2 += (A2[i, j] - prev_A2[i, j])^2
            sum_a2 += (A2[i, j] + prev_A2[i, j])^2
        end # for i

        dev_s = sum_s = zero(T)
        for i in axes(S, 2)
            dev_s += (S[j, i] - prev_S[j, i])^2
            sum_s += (S[j, i] + prev_S[j, i])^2
        end # for i

        devmax = max(devmax, sqrt(max(dev_a2/sum_a2, dev_s/sum_s)))
        if sqrt(dev_a2) > eps*sqrt(sum_a2) || sqrt(dev_s) > eps*sqrt(sum_s)
            return false, devmax
        end # if
    end # for j

    return true, devmax
end # function check_convergence
