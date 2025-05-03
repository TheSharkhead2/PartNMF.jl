using LinearAlgebra

struct MultUpdate{T} <: NMFAlgorithm{T}
    maxiter::Integer
    ε::Real
    tol::Real

    function MultUpdate{T}(;
        maxiter::Integer=100,
        ε::Real=eps(T),
        tol::Real=cbrt(eps(T))
    ) where T
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be postive."))
        ε > 0 || throw(ArgumentError("ε must be positive."))

        new{T}(maxiter, ε, tol)
    end # function MultUpdate
end # struct MultUpdate

struct MultUpdate_State{T}
    XSt::Matrix{T}
    SSt::Matrix{T}
    A1SSt::Matrix{T}
    A2SSt::Matrix{T}
    A1tX::Matrix{T}
    A2tX::Matrix{T}
    A2tA1::Matrix{T}
    A1tA2::Matrix{T}
    A1tA1::Matrix{T}
    A2tA2::Matrix{T}
    A2tA1S::Matrix{T}
    A1tA2S::Matrix{T}
    A1tA1S::Matrix{T}
    A2tA2S::Matrix{T}

    function MultUpdate_State{T}(
        X::Matrix{T},
        A1::Matrix{T},
        A2::Matrix{T},
        S::Matrix{T}
    ) where T
        n, l, k = checksize(X, A1, A2, S)

        A1t = transpose(A1)
        new{T}(
            Matrix{T}(undef, n, l),
            Matrix{T}(undef, l, l),
            Matrix{T}(undef, n, l),
            Matrix{T}(undef, n, l),
            A1t * X, # constant during optimization
            Matrix{T}(undef, l, k),
            Matrix{T}(undef, l, l),
            Matrix{T}(undef, l, l),
            A1t * A1, # constant during optimization
            Matrix{T}(undef, l, l),
            Matrix{T}(undef, l, k),
            Matrix{T}(undef, l, k),
            Matrix{T}(undef, l, k),
            Matrix{T}(undef, l, k),
        )
    end # function MultUpdate_State
end # struct MultUpdate_State{T}

prepare_state(::MultUpdate{T}, X, A1, A2, S) where T = MultUpdate_State{T}(X, A1, A2, S)

function update!(
    alg::MultUpdate{T},
    state::MultUpdate_State{T},
    X::Matrix{T},
    A1::Matrix{T},
    A2::Matrix{T},
    S::Matrix{T}
) where T
    # unwrap state
    ε = alg.ε
    XSt = state.XSt
    SSt = state.SSt
    A1SSt = state.A1SSt
    A2SSt = state.A2SSt
    A1tX = state.A1tX
    A2tX = state.A2tX
    A2tA1 = state.A2tA1
    A1tA2 = state.A1tA2
    A1tA1 = state.A1tA1
    A2tA2 = state.A2tA2
    A2tA1S = state.A2tA1S
    A1tA2S = state.A1tA2S
    A1tA1S = state.A1tA1S
    A2tA2S = state.A2tA2S

    # update A_2
    St = transpose(S)
    mul!(XSt, X, St)
    mul!(SSt, S, St)
    mul!(A1SSt, A1, SSt)
    mul!(A2SSt, A2, SSt)
    
    # TODO: @turbo
    @inbounds for i in 1:length(A2)
        A2[i] *= XSt[i] / (A1SSt[i] + A2SSt[i] + ε)
    end # for i

    # update S
    A1t = transpose(A1)
    A2t = transpose(A2)
    mul!(A2tX, A2t, X)
    mul!(A2tA1, A2t, A1)
    mul!(A1tA2, A1t, A2)
    mul!(A2tA2, A2t, A2)
    mul!(A2tA1S, A2tA1, S)
    mul!(A1tA2S, A1tA2, S)
    mul!(A1tA1S, A1tA1, S)
    mul!(A2tA2S, A2tA2, S)

    # TODO: @turbo
    @inbounds for i in 1:length(S)
        S[i] *= (A1tX[i] + A2tX[i]) / (
            A2tA1S[i] + A1tA2S[i] + A1tA1S[i] + A2tA2S[i] + ε
        ) 
    end # for i
end # function update!
