# -*- coding: utf-8 -*-
function compute_jump_probabilities(V_t::Float64,r::Float64,Δx::Float64,D::Float64)
    Z = (1/4) * (V_t * Δx) / D
    P = 1-r
    P⁺ = (exp(Z))/(exp(Z) + exp(-Z))*r
    P⁻ = 1 - P - P⁺
    
    return P::Float64, P⁺::Float64, P⁻::Float64, V_t::Float64 
end

function randomness_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64,t::Int64; t_current::Int64=-1)
    slob = D[slob_num].slob
    rn = slob.randomness_term
    V_list = D[slob_num].V
    
    if (rn.do_random)
        if t_current != -1
            t = t_current
        end
        
        if (rn.do_random_walk)
            ϵ = rand(Normal(0.0,rn.σ))# changed recently
            #ϵ = rand()*2 - 1# changed recently
            
            prev_V = V_list[t-1]
            #center = 0
            #V_t = center + rn.β*(prev_V-center) + ϵ * rn.σ  #sign(ϵ) * min(  abs(rn.σ * ϵ)  ,   slob.Δx / slob.Δt  ) # Ensures V_t¹ = ϵ¹σ ≤ Δx/Δt  ????
            V_t = rn.β*prev_V + ϵ #* rn.σ  #sign(ϵ) * min(  abs(rn.σ * ϵ)  ,   slob.Δx / slob.Δt  ) # Ensures V_t¹ = ϵ¹σ ≤ Δx/Δt  ????
        else 
            if ((rand()<rn.β) && (t>rn.lag+1))
                V_t = V_list[t-rn.lag]
            else
                ϵ = rand(Normal(0.0,rn.σ))
                #ϵ = rand()*2 - 1# changed recently
                V_t = sign(ϵ) * min(  abs(rn.σ * ϵ)  ,   slob.Δx / slob.Δt  ) # Ensures V_t¹ = ϵ¹σ ≤ Δx/Δt  ????
            end
        end
        
        # Z = (1/4) * (V_t * slob.Δx) / slob.D
        # P = 1-rn.r
        # P⁺ = (exp(Z))/(exp(Z) + exp(-Z))*rn.r
        # P⁻ = 1 - P - P⁺
        return compute_jump_probabilities(V_t,rn.r,slob.Δx,slob.D)
        
    else # for speed. Could have just called the above function with V_t=0
        P = 1-rn.r
        P⁺ = (rn.r)/2
        P⁻ = P⁺
        V_t = 0.0
    end
    
    
    return P::Float64, P⁺::Float64, P⁻::Float64, V_t::Float64
end
