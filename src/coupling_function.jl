# -*- coding: utf-8 -*-
function CouplingTerm(μ::Float64, a::Float64, b::Float64, c::Float64)
    return CouplingTerm(μ, a, b, c, true)
end

# +
function coupling_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1)
    
    it = D[slob_num].slob.coupling_term
    if (!(it.do_coupling) || it.b==0.0)
        #return D[slob_num].slob.zero_vec #NB passes by sharing
        return D[slob_num].slob.x .* 0 #NB passes by sharing
    end
    
    
    # extract most recent prices
    #p¹ = p_list¹[t-1]
    #p² = p_list²[t-1]
    
    p¹ = D[slob_num].raw_price_paths[t-1]
    p² = D[2+(1-slob_num)].raw_price_paths[t-1]
    
    function coupling_inner(x, p¹, p², t) 
        f(y)=-(it.μ*(y))*exp(-(it.μ*(y))^2) #y is a temporary variable
        g = 1+tanh(abs(p²-p¹)/it.a)*it.c

        if (p²>p¹) 
            if x>p¹
                return it.b* f(1/g*(x-p¹))
            else 
                return it.b* g*f(x-p¹)      
            end
        else
            if x<p¹
                return it.b* f(1/g*(x-p¹))
            else 
                return it.b* g*f(x-p¹)
            end
        end
    end
    
    coupling¹ = [coupling_inner(xᵢ¹, p¹, p², t) for xᵢ¹ in D[slob_num].slob.x]
    
end
