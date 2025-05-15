using PartNMF
using Test
using Random
using LinearAlgebra

tests = [
    "interface"
]

@testset "All tests" begin
    for t in tests
        include("$t.jl")
    end # for t
end # @testset "All tests" begin
