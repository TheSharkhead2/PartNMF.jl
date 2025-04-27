
function randinit(X::AbstractMatrix{T}, l::Integer) where T
    n, k = size(X)

    A2 = rand(T, n, l)
    S = rand(T, l, k)

    return tuple(A2, S)
end # function randinit
