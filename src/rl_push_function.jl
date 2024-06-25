# -*- coding: utf-8 -*-
# +
function rl_push_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1)
    
    
    slob = D[slob_num].slob
    rl = slob.rl_push_term
    
    
    #return temp # return while its still 0 for no pushing 
    
    if rl.Do&&(t>=rl.StartTime)&&(t<rl.EndTime) #never run with t=1
        temp = slob.zero_vec[:] ####SUPER NB, avoid pass by sharing as we are about to modify
        
        if (rl.Position>0)
            temp[rl.Position] = -rl.Amount
        else 
            my_p = price_to_index(D[slob_num].raw_price_paths[t-1], slob.Δx, slob.x[1])
            
            index_to_modify = my_p - rl.Position
            
            temp[index_to_modify] = temp[index_to_modify] - rl.Amount
        end
    else
        temp = slob.zero_vec
    end
    
    return temp
    
end