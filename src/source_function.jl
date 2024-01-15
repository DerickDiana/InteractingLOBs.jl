# -*- coding: utf-8 -*-
# +
function source_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1)
    
    st = D[slob_num].slob.source_term
    
    if !(st.do_source)
        return D[slob_num].slob.zero_vec #NB passes by sharing
    end
    
    p¹ = D[slob_num].raw_price_paths[t-1]
    
    f(y)=-st.λ*st.μ*y*exp(-(st.μ*(y))^2) #y is a temporary variable
    #f(y)=-10*sign(y) #y is a temporary variable
    #f(y) = -st.λ*tanh(st.μ*(y))
    
    width = D[slob_num].slob.L
    
    return [f(  mod(  xᵢ¹ - p¹ - width/2  , width  )  -  width/2  ) for xᵢ¹ in D[slob_num].slob.x]
    
end
