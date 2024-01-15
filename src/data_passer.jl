# -*- coding: utf-8 -*-
mutable struct DataPasser
    slob::SLOB
    lob_densities::Matrix{Float64}
    lob_densities_L::Matrix{Float64}
    lob_densities_R::Matrix{Float64}
    sources::Matrix{Float64}
    couplings::Matrix{Float64}  
    rl_pushes::Matrix{Float64}
    raw_price_paths::Vector{Float64}
    obs_price_paths::Vector{Float64}
    P⁺s::Vector{Float64}
    P⁻s::Vector{Float64}
    Ps::Vector{Float64} 
    V::Vector{Float64}
    x_shifts::Vector{Float64}
    
    RLBrains::Vector{RLBrain}
    RLParam::RLParam
    RLViews::Vector{RLView}
end

function DataPasser(slob::SLOB,
    lob_densities::Matrix{Float64},
    lob_densities_L::Matrix{Float64},
    lob_densities_R::Matrix{Float64},
    sources::Matrix{Float64},
    couplings::Matrix{Float64},
    rl_pushes::Matrix{Float64},
    raw_price_paths::Vector{Float64},
    obs_price_paths::Vector{Float64},
    P⁺s::Vector{Float64},
    P⁻s::Vector{Float64},
    Ps::Vector{Float64},
    V::Vector{Float64},
    x_shifts::Vector{Float64})
    
    return DataPasser(slob,
    lob_densities,
    lob_densities_L,
    lob_densities_R,
    sources,
    couplings,
    rl_pushes,
    raw_price_paths,
    obs_price_paths,
    P⁺s,
    P⁻s,
    Ps,
    V,
    x_shifts,
    Array{RLBrain}(undef,0),
    RLParam(),
    Array{RLView}(undef,0)
    )
end
