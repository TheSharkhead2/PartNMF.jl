struct MultUpdate{T}
    maxiter::Integer
    ε::Real
    tol::Real

    function MultUpdate{T}(;
        maxiter::Integer=100,
        ε=eps(T),
        tol::Real=cbrt(eps(T))
    )
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be postive."))
        ε > 0 || throw(ArgumentError("ε must be positive."))

        new{T}(maxiter, ε, tol)
    end # function MultUpdate
end # struct MultUpdate


