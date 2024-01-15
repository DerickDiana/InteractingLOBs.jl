# -*- coding: utf-8 -*-
struct SourceTerm
    λ::Float64
    μ::Float64
    do_source::Bool
end


struct CouplingTerm
    μ::Float64
    a::Float64
    b::Float64
    c::Float64
    do_coupling::Bool
end

struct RLPushTerm
        StartTime::Int
        EndTime::Int
        Position::Int
        Amount::Float64
        Do::Bool
        RemovalType::Bool
end

struct RandomnessTerm
        σ::Float64   # Standard deviation in whatever distribution
        r::Float64   # Probability of self jump
        β::Float64   #probability of being the same as value lag ago
        lag::Int64 #lag position
        do_random_walk::Bool #probability of behaving like a random walk
        do_random::Bool
end
