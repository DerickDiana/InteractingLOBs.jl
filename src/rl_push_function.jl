# -*- coding: utf-8 -*-
mutable struct RLPushTerm
        StartTime::Int
        EndTime::Int
        Position::Int
        Amount::Float64
        Do::Bool
end

# +
function (rl::RLPushTerm)(slob¹, φ_list¹, p_list¹, 
                          slob², φ_list², p_list², 
                          t)
    
    temp = [0.0 for xᵢ¹ in slob¹.x] 
    
    # return temp # return while its still 0 for no pushing 
    
    if rl.Do&&(t>=rl.StartTime)&&(t<rl.EndTime) #never run with t=1
        if (rl.Position>0)
            #temp[rl.Position] = -rl.Amount
        else 
            latest_φ = φ_list¹[:,t-1]
            
            my_p = extract_mid_price_index(slob¹,latest_φ)
            
            index_to_modify = my_p - rl.Position
            
            left_side = [1:my_p;]
            
            # optional rescale of all points except index_to_modify (it is overwritten below) 
            # to have negative the area that index_to_modify has
            
            area = sum(latest_φ[left_side])
            rescaled_latest_φ = latest_φ[left_side] ./ area .* rl.Amount
            temp[left_side] = rescaled_latest_φ
            
            
            temp[index_to_modify] = temp[index_to_modify] - rl.Amount
            
        end
    end
    
    
    
    return temp
    
end
# -

a = [1:5;]

a[1:end .!= 3]


