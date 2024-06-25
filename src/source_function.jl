# -*- coding: utf-8 -*-
# +
function source_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1)
    
    st = D[slob_num].slob.source_term
    μ = D[slob_num].temp[1]
    
    if !(st.do_source)
        return D[slob_num].slob.zero_vec #NB passes by sharing
    end
    
    p¹ = D[slob_num].raw_price_paths[t-1]
    
    f(y)=y
    
    ################## Track a random walk
    alpha = 0.1
    μ = 0.7 + alpha * (μ - 0.7) + rand(Normal(0.0,0.3))

    D[slob_num].temp[1] = μ
    
    # All functions f below expect the price intercept to occur at the point where the input (y) is 0
    ################## Normal constant source
    if false
        f(y)=-st.λ*st.μ*y*exp(-(st.μ*(y))^2) 
    end
    
    ################# Changing scale on random walk:
    if false
        f(y)=-st.λ*μ*y*exp(-(μ*(y))^2) 
    end
    
    ################# Changing angle on random walk:
    if true
        f = function f(y)
            if y>0
                μ2 = 0.7 - (μ-0.7)
            else
                μ2 = μ
            end

            -st.λ*μ2*y*exp(-(μ2*(y))^2) 
        end
    end
    
    
    #f(y)=-10*sign(y) #y is a temporary variable
    #f(y) = -st.λ*tanh(st.μ*(y))
    
    width = D[slob_num].slob.L
    
    return [f(  mod(  xᵢ¹ - p¹ - width/2  , width  )  -  width/2  ) for xᵢ¹ in D[slob_num].slob.x]
    
end
# -




