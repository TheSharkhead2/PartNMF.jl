using Statistics
using RandomizedLinAlg

function randinit(X::AbstractMatrix{T}, l_2::Integer, l::Integer) where T
    n, k = size(X)

    A2 = rand(T, n, l_2)
    S = rand(T, l, k)

    return tuple(A2, S)
end # function randinit

# adapted from [NMF.jl](https://github.com/JuliaStats/NMF.jl) 
function _nndsvd!(U, Σ, V, X, A2, St_small, initS::Bool, variant::Symbol)
    l_2 = size(A2, 2)
    T = eltype(A2)

    U = T.(U)
    Σ = T.(Σ)
    V = T.(V)

    v0 = variant == :std ? zero(T) :
        (variant == :a   ? convert(T, mean(X)) : convert(T, mean(X) * 0.01))

    for j in 1:l_2
        x = view(U, :, j)
        y = view(V, :, j)
        xpnrm, xnnrm = posnegnorm(x)
        ypnrm, ynnrm = posnegnorm(y)

        mp = xpnrm * ypnrm
        mn = xnnrm * ynnrm

        vj = v0
        if variant == :ar
            vj *= rand(T)
        end # if

        if initS
            if mp >= mn
                ss = sqrt(Σ[j] * mp)
                scalepos!(view(A2, :, j), x, ss / xpnrm, vj)
                scalepos!(view(St_small, :, j), y, ss / ypnrm, vj)
            else
                ss = sqrt(Σ[j] * mn)
                scaleneg!(view(A2, :, j), x, ss / xnnrm, vj)
                scaleneg!(view(St_small, :, j), y, ss / ynnrm, vj)
            end # if
        else
            if mp >= mn
                ss = sqrt(Σ[j] * mp)
                scalepos!(view(A2, :, j), x, ss / xpnrm, vj)
            else
                ss = sqrt(Σ[j] * mn)
                scaleneg!(view(A2, :, j), x, ss / xnnrm, vj)
            end # if
        end # if
    end # for j
end # function _nndsvd!

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

    A2 = Matrix{T}(undef, n, l_2)
    S_small = Matrix{T}(undef, l_2, k) # should be l, but easier to ignore extra dim

    if constantS
        St_small = reshape(view(S_small, :, :), (k, l_2))
        _nndsvd!(U, Σ, V, X, A2, St_small, false, variant)
        fill!(S_small, 1/l)
    else
        St_small = Matrix{T}(undef, k, l_2)
        _nndsvd!(U, Σ, V, X, A2, St_small, true, variant)
        for j in 1:l_2
            for i in 1:k
                S_small[j, i] = St_small[i, j]
            end # for i
        end # for j
    end # if

    # expand S matrix to correct size (fill empty values with 1/l)
    S = vcat(fill(T(1/l), (l - l_2, k)), S_small)

    return (A2, S)
end # function nndsvd

# adapted from [NMF.jl](https://github.com/JuliaStats/NMF.jl) 
function posnegnorm(x::AbstractArray{T}) where T
    pn = zero(T)
    nn = zero(T)
    for i in eachindex(x)
        @inbounds xi = x[i]
        if xi > zero(T)
            pn += abs2(xi)
        else
            nn += abs2(xi)
        end # if
    end # for i

    return (sqrt(pn), sqrt(nn))
end # function posnegnorm

# adapted from [NMF.jl](https://github.com/JuliaStats/NMF.jl) 
function scalepos!(y, x, c::T, v0::T) where T<:Number
    @inbounds for i in eachindex(y)
        xi = x[i]
        if xi > zero(T)
            y[i] = xi * c
        else
            y[i] = v0
        end
    end
end

# adapted from [NMF.jl](https://github.com/JuliaStats/NMF.jl) 
function scaleneg!(y, x, c::T, v0::T) where T<:Number
    @inbounds for i in eachindex(y)
        xi = x[i]
        if xi < zero(T)
            y[i] = - (xi * c)
        else
            y[i] = v0
        end
    end
end
