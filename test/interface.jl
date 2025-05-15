@testset "interface" begin
    n = 12
    l_1 = 2
    l_2 = 3
    k = 8

    for T in (Float64, Float32)
        A1 = max.(rand(T, n, l_1) .- T(0.3), zero(T))
        A2g = max.(rand(T, n, l_2) .- T(0.3), zero(T))
        Sg = max.(rand(T, l_1 + l_2, k) .- T(0.3), zero(T))
        X = hcat(A1, A2g) * Sg

        for alg in (:mult,)
            for init in (:rand,)
                ret = PartNMF.part_nmf(X, A1, l_1 + l_2; alg=alg, init=init)
            end # for init
        end # for alg
    end # for T
end # @testset "interface" begin
