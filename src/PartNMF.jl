module PartNMF

export part_nmf

include("common.jl")
include("initialization.jl")
include("mult_part_nmf.jl")
include("interface.jl")

# precompilation workload to avoid precompiling on first run
# note, this is taken from NMF.jl
using PrecompileTools

let 
    @setup_workload begin
        X = rand(8, 6)
        A1 = rand(8, 2)
        @compile_workload begin
            for alg in (:mult,)
                for init in (:rand, :nndsvd, :nndsvda, :nndsvdar)
                    part_nmf(X, A1, 4, alg=alg, init=init)
                end # for init
            end # for alg
        end # @compile_workload begin
    end # @setup_workload begin
end # let

end # module PartNMF
