
function randinit(X::AbstractMatrix{T}, l_2::Integer, l::Integer) where T
    n, k = size(X)

    A2 = rand(T, n, l_2)
    S = rand(T, l, k)

    return tuple(A2, S)
end # function randinit
