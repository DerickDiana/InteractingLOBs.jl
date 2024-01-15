# -*- coding: utf-8 -*-
function special_function(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1, move_price_with=false)
    if true && t > 2
        ####Catch AI info
        RLParam_ = D[slob_num].RLParam
        
        RLBrain_ = D[slob_num].RLBrains[t-1]
        RLView_  = D[slob_num].RLViews[t-1]
        
        x_s = RLView_.x_s
        
        if (x_s[1] < 0) #return 0 end #end the training
            l = length(D[slob_num].RLBrains)
            
            if RLBrain_ == D[slob_num].RLBrains[l]
                return 0
            end
            
            
            D[slob_num].RLBrains[t:l] = repeat([RLBrain_],l-t+1)
            D[slob_num].RLViews[t:l] = repeat([RLView_],l-t+1)
            
            return 0
        end

        #####Update AI here
        f,RLBrain__ = ContinuousLearning.updatelearningAC(RLView_,RLBrain_,RLParam_)
        
        #f = (rand()-0.5) * 0.05
        f = f/100
        #going_left = f < 0
        going_left = true
        f = abs(f)
        
        #####Make AI do market order here
        do_market_order_using_area(D,slob_num,t,f;t_current=t_current,move_price_with=move_price_with,
                                         going_left=going_left,
                                         take_grad_mid_point=false,take_mid_point=true,do_smoothing=true)
        
        r = (D[slob_num].raw_price_paths[t-1] - D[slob_num].slob.p₀)/(D[slob_num].slob.L)
        
        ####Pass AI info
        D[slob_num].RLBrains[t] = RLBrain__
        D[slob_num].RLViews[t] = RLView([x_s[1]-f/3,t/D[slob_num].slob.N],r)
        #print(D[slob_num].RLViews[t].x_s[1]," ",D[slob_num].RLViews[t].x_s[2]," ",D[slob_num].RLViews[t].r,"\n")

    end

    if false
        #####Do normal market kick
        slob = D[slob_num].slob
        rl = slob.rl_push_term

        if (!rl.Do)&&(t>=rl.StartTime+2)&&(t<rl.EndTime+2) && rl.RemovalType #never run with t=1

            do_market_order_using_area(D,slob_num,t,rl.Amount;t_current=t_current,move_price_with=move_price_with,
                                         going_left=!true,take_grad_mid_point=false,take_mid_point=true,do_smoothing=true)

        end
    end
    
    # 0.1, 1% of the time
    if t>20 && false && rand()<0.3
        r_f = (rand()-0.5) * 0.01
        #r_f = D[slob_num].V[t-1]/10
        going_left = r_f < 0
        r_f = abs(r_f)
            
        do_market_order_using_area(D,slob_num,t,r_f;t_current=t_current,move_price_with=move_price_with,
                                     going_left=going_left,take_grad_mid_point=false,take_mid_point=true,do_smoothing=true)
        
    end
end

function do_market_order_using_area(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64, amount::Float64; t_current::Int64=-1, move_price_with=false, going_left=true, take_grad_mid_point=true, take_mid_point=true, do_smoothing=true,debug_print=false)
    slob = D[slob_num].slob
    rl = slob.rl_push_term
    
    #return temp # return while its still 0 for no pushing 
    do_trapezium = true
    
        
    φ = D[slob_num].lob_densities[:,t-1]
    l = length(φ)

    my_p = price_to_index(D[slob_num].raw_price_paths[t-1], slob.Δx, slob.x[1]) + 1 #This picks out the point to the the left of where the intercept is but only with the +1

    # i.e.:
    #    +         +/0           -
    # my_p-1      my_p       my_p + 1
    #    A          B            C

    if going_left  #if we're going left
        dir = -1  # point left

        if φ[my_p] != 0.0 #note that if this happens, you will get my_p included in the list positions_to_set_to_0 below
            start_pos = my_p + 1  #move from B to C 
        else
            start_pos = my_p
        end

    else 
        dir = +1  # point right

        start_pos = my_p #stay at B
    end
    
    
    if debug_print print("Start pos: ",start_pos,"\n") end

    sum_so_far = 0    #  We haven't encountered anything of the opposite sign yet

    curr_pos = start_pos   # We're going to keep track of the current position starting from the start position

    curr_pos += dir                                     # move in the chosen direction

    frac = calculate_intercept(φ[start_pos],φ[curr_pos],0;Δx=slob.Δx)/slob.Δx
    closest_zero = slob.x[start_pos] + dir * frac * slob.Δx
    
    if debug_print print("Closest zero: ",closest_zero,"\n") end

    potential_next_sum = sum_so_far+abs(calculate_trapezium_area(0,φ[curr_pos],(1-frac)*slob.Δx))
    
    
    if debug_print print("Potential next sum:",potential_next_sum,"\n") end
    
                                                        # Now you should for the first time encounter some density that you can start to use the fulfill the order. 
                                                        # Record how much you could fulfill if you include this order

    while potential_next_sum < amount # Is the potential inclusion of the density you're on enough to fulfill the order?

        sum_so_far = potential_next_sum  # If it is not, then you will need to record that you use all the density avaliable at this point
        
        if debug_print print("Sum so far: ",sum_so_far,"\n") end

        curr_pos += dir                  # then move to the next point

        if curr_pos == 1 || curr_pos == l
            @warn "You may have made a market order larger than the total area of the LOB"
        end

        potential_next_sum = sum_so_far+abs(calculate_trapezium_area(φ[curr_pos-dir],φ[curr_pos],slob.Δx))
                                         # Record how much you could fulfill if you include the density you're on
    end
    
    if debug_print print("Indices moved: ",start_pos - curr_pos,"\n") end

    positions_to_set_to_0 = (start_pos+dir):dir:(curr_pos-dir) # check which positions you completely used to fulfill the order and set them to 0
    if debug_print print("Indices set to zero: ",positions_to_set_to_0,"\n") end
    
    for i in  positions_to_set_to_0             # now loop over all the positions  
                                                # (not including where you started i.e. start_pos+dir or stopped i.e. curr_pos-dir)
        D[slob_num].lob_densities[i,t-1] = 0.0  # set all these to 0
        if slob.do_exp_dist_times
            D[slob_num].lob_densities_L[i+1,t-1] = 0.0  # set all these to 0 #was -
            D[slob_num].lob_densities_R[i-1,t-1] = 0.0  # set all these to 0 #was +
        else
            D[slob_num].lob_densities_L[i,t-1] = 0.0  # set all these to 0
            D[slob_num].lob_densities_R[i,t-1] = 0.0  # set all these to 0
        end
    end

    

    if curr_pos == start_pos + dir
        farthest_zero = slob.x[start_pos] + dir * frac * slob.Δx + 
                        dir * calculate_distance_to_get_volume(0,-dir*φ[curr_pos],slob.Δx*(1-frac),amount)

        p = amount/calculate_trapezium_area(0,-dir*φ[curr_pos],(1-frac)*slob.Δx)
    else
        farthest_zero = slob.x[curr_pos-dir] +
                        dir * calculate_distance_to_get_volume(-dir*φ[curr_pos-dir],-dir*φ[curr_pos],slob.Δx,amount-sum_so_far)
        if debug_print print("Farthest zero: ",farthest_zero,"\n") end

        p = (amount-sum_so_far)/calculate_trapezium_area(-dir*φ[curr_pos-dir],-dir*φ[curr_pos],slob.Δx)
        
        if debug_print print("p: ",p,"\n") end
    end

    D[slob_num].raw_price_paths[t-1] = (farthest_zero + closest_zero)/2.0

    if do_smoothing
        new_val = (1 - p) * φ[curr_pos]
        D[slob_num].lob_densities[curr_pos,t-1] = new_val
        if slob.do_exp_dist_times
            D[slob_num].lob_densities_L[curr_pos+1,t-1] = new_val #was -
            D[slob_num].lob_densities_R[curr_pos-1,t-1] = new_val #was +
        else
            D[slob_num].lob_densities_L[curr_pos,t-1] = new_val
            D[slob_num].lob_densities_R[curr_pos,t-1] = new_val
        end
    end
end

# +
function do_market_order(D::Dict{Int64, DataPasser},
                                 slob_num::Int64, t::Int64; t_current::Int64=-1, move_price_with=false, going_left=true, take_grad_mid_point=true, take_mid_point=true, do_smoothing=true)
    slob = D[slob_num].slob
    rl = slob.rl_push_term
    
    #return temp # return while its still 0 for no pushing 
    
    if (!rl.Do)&&(t>=rl.StartTime+2)&&(t<rl.EndTime+2) && rl.RemovalType #never run with t=1
        
        φ = D[slob_num].lob_densities[:,t-1]

        my_p = price_to_index(D[slob_num].raw_price_paths[t-1], slob.Δx, slob.x[1]) + 1 #This picks out the point to the the left of where the intercept is but only with the +1
        
        # i.e.:
        #    +         +/0           -
        # my_p-1      my_p       my_p + 1
        #    A          B            C
        
        # for proof:
        # print(my_p-1," ",round(φ[my_p-1];digits=2)," ",my_p," ",round(φ[my_p];digits=2)," ",my_p+1," ",round(φ[my_p+1];digits=2),"\n")

        
        if going_left  #if we're going left
            dir = -1  # point left
            start_pos = my_p + 1  #move from B to C 
        else 
            dir = +1  # point right
            
            if φ[my_p] == 0.0 #note that if this happens, you will get my_p included in the list positions_to_set_to_0 below
                start_pos = my_p - 1  #move from B to A
            else
                start_pos = my_p #stay at B
            end
        end
        
        sum_so_far = 0    #  We haven't encountered anything of the opposite sign yet
        
        curr_pos = start_pos   # We're going to keep track of the current position starting from the start position
        
        curr_pos += dir                                     # move in the chosen direction
        potential_next_sum = sum_so_far+abs(φ[curr_pos])    # Now you should for the first time encounter some density that you can start to use the fulfill the order. 
                                                            # Record how much you could fulfill if you include this order
        
        while potential_next_sum < rl.Amount # Is the potential inclusion of the density you're on enough to fulfill the density?

            sum_so_far = potential_next_sum  # If it is not, then you will need to record that you use all the density avaliable at this point
            
            curr_pos += dir                  # then move to the next point
            potential_next_sum = sum_so_far+abs(φ[curr_pos]) # Record how much you could fulfill if you include the density you're on
        end

        positions_to_set_to_0 = (start_pos+dir):dir:(curr_pos-dir) # check which positions you completely used to fulfill the order and set them to 0
        for i in  positions_to_set_to_0              # now loop over all the positions  
                                                    # (not including where you started i.e. start_pos+dir or stopped i.e. curr_pos-dir)
            D[slob_num].lob_densities[i,t-1] = 0.0  # set all these to 0
        end

        new_final_value = φ[curr_pos]+dir*(rl.Amount-sum_so_far) # Subtract what remains of the order from the position at which you stopped
        D[slob_num].lob_densities[curr_pos,t-1] = new_final_value
        
        if move_price_with
            D[slob_num].raw_price_paths[t-1] = index_to_price((curr_pos+1)-1, slob.Δx, slob.x[1])  # Set the price to the point at which you stopped
                                                                                                   # -1 is here for now as I still need to correct the function itself
        end
        
        if take_grad_mid_point 
            y1 = new_final_value #y value (density) to the left
            y2 = φ[start_pos]   #y value (density) to the right
            #x1 = slob.x[start_pos]      #x value value to the left
            x2 = slob.x[start_pos]      #x value value to the right


            #mid_price = (-y1 * slob.Δx)/(y2 - y1) + x1 
            mid_price = (-y2 * (start_pos-curr_pos)*slob.Δx)/(y2 - y1) + x2 
            
            D[slob_num].raw_price_paths[t-1] = mid_price
        end
        
        if take_mid_point
            
            if length(positions_to_set_to_0) == 0 # if you didn't set anything to 0. Can only happen if φ[my_p] != 0.0
                y1 = new_final_value              # y value (density) to the left
                y2 = φ[start_pos]                 # y value (density) to the right
                #x1 = slob.x[start_pos]           # x value value to the left
                x2 = slob.x[start_pos]            # x value value to the right

                #mid_price = (-y1 * slob.Δx)/(y2 - y1) + x1 
                mid_price = (-y2 * (start_pos-curr_pos)*slob.Δx)/(y2 - y1) + x2 
            else
                closest_zero_index = maximum(positions_to_set_to_0)
                farthest_zero_index = minimum(positions_to_set_to_0)
                
                closest_zero_x     = slob.x[closest_zero_index]
                farthest_zero_x    = slob.x[farthest_zero_index]
                
                if do_smoothing #&& (length(positions_to_set_to_0)>1)
                    
                    beyond_farthest_zero_x = slob.x[farthest_zero_index + dir]
                    
                    old_mid_price    = (closest_zero_x + farthest_zero_x              )/2
                    target_mid_price = (closest_zero_x + beyond_farthest_zero_x       )/2
                    
                    frac_of_way = (φ[curr_pos]-new_final_value)/(φ[curr_pos])
                    
                    mid_price = (1-frac_of_way) * old_mid_price + frac_of_way * target_mid_price
                else
                    mid_price = (closest_zero_x + farthest_zero_x)/2
                end
            end
            
            D[slob_num].raw_price_paths[t-1] = mid_price
            
        end
        
    end
    
end
# -

function calculate_area(φ_interp,x_dense,left_x,right_x)
    left_ind  = findfirst((x) -> x > left_x,  x_dense)
    right_ind = findfirst((x) -> x > right_x, x_dense)
    
    return sum(φ_interp[left_ind:right_ind]) * (x_dense[2] - x_dense[1])
end

function calculate_distance_to_get_volume_smooth(φ_interp,x_dense,target_volume,right_x)
    sum_so_far = 0
    
    right_ind = findfirst((x) -> x > right_x, x_dense) 
    delta_x = x_dense[2] - x_dense[1] 
    
    i = right_ind
    
    while sum_so_far < target_volume
        sum_so_far += delta_x * φ_interp[i]
        i -= 1
    end
    
    return x_dense[i]
end

function calculate_intercept(y1,y2,x1;Δx=0.0,x2=Inf)
    if Δx==0
        @assert x2 != Inf
        Δx = x2 - x1
    end
    
    return (-y1 * Δx)/(y2 - y1) + x1 
end

function calculate_distance_to_get_volume(y1,y2,Δx,V)
    if y1==y2
        return V/y1
    else
        M = (y2 - y1)/(2*Δx)
        return (-y1 + sqrt(y1^2 + 4 * M * V))/(2*M)  
    end
end

function calculate_trapezium_area(y1,y2,Δx)
   return (y1+y2)/2 * Δx
end

function calculate_trapezium_area(y1,y2,x1,x2)
   return (y1+y2)/2 * abs(x2-x1)
end

function calculate_trapezium_area_many(φ,Δx,left_ind,right_ind)
    sums = []
    for i in left_ind:(right_ind-1)
        push!(sums,calculate_trapezium_area(φ[i],φ[i+1],Δx))
    end
    return reverse(sums)
end

# +
# function do_market_order_using_area(D::Dict{Int64, DataPasser},
#                                  slob_num::Int64, t::Int64, amount::Float64; t_current::Int64=-1, move_price_with=false, going_left=true, take_grad_mid_point=true, take_mid_point=true, do_smoothing=true)
#     slob = D[slob_num].slob
#     rl = slob.rl_push_term
    
#     #return temp # return while its still 0 for no pushing 
#     do_trapezium = true
    
        
#     φ = D[slob_num].lob_densities[:,t-1]
#     l = length(φ)

#     my_p = price_to_index(D[slob_num].raw_price_paths[t-1], slob.Δx, slob.x[1]) + 1 #This picks out the point to the the left of where the intercept is but only with the +1

#     # i.e.:
#     #    +         +/0           -
#     # my_p-1      my_p       my_p + 1
#     #    A          B            C

#     # for proof:

#     #print(my_p-1," ",round(φ[my_p-1];digits=2)," ",my_p," ",round(φ[my_p];digits=2)," ",my_p+1," ",round(φ[my_p+1];digits=2),"\n")


#     if going_left  #if we're going left
#         dir = -1  # point left

#         if φ[my_p] != 0.0 #note that if this happens, you will get my_p included in the list positions_to_set_to_0 below
#             start_pos = my_p + 1  #move from B to C 
#         else
#             start_pos = my_p
#         end

#     else 
#         dir = +1  # point right

#         start_pos = my_p #stay at B
#     end

#     sum_so_far = 0    #  We haven't encountered anything of the opposite sign yet

#     curr_pos = start_pos   # We're going to keep track of the current position starting from the start position

#     curr_pos += dir                                     # move in the chosen direction

#     if do_trapezium
#         frac = calculate_intercept(φ[start_pos],φ[curr_pos],0;Δx=slob.Δx)/slob.Δx
#         closest_zero = slob.x[start_pos] + dir * frac * slob.Δx

#         potential_next_sum = sum_so_far+calculate_trapezium_area(0,φ[curr_pos],(1-frac)*slob.Δx)
#     else
#         closest_zero = get_interpolated_intercept(slob.x_range,φ,slob.x[curr_pos],slob.x[start_pos])
#         frac = abs(slob.x[curr_pos] - closest_zero)/slob.Δx

#         x_dense = range(slob.x_range[1],slob.x_range[end], length = length(slob.x_range) * 500)
#         φ_interp = auto_interpolator(slob.x_range,φ,x_dense)


#         potential_next_sum = sum_so_far+calculate_area(φ_interp,x_dense,slob.x[curr_pos],closest_zero)
#     end

#                                                         # Now you should for the first time encounter some density that you can start to use the fulfill the order. 
#                                                         # Record how much you could fulfill if you include this order

#     while potential_next_sum < amount # Is the potential inclusion of the density you're on enough to fulfill the order?

#         sum_so_far = potential_next_sum  # If it is not, then you will need to record that you use all the density avaliable at this point

#         curr_pos += dir                  # then move to the next point

#         if curr_pos == 1 || curr_pos == l
#             @warn "You may have made a market order larger than the total area of the LOB"
#         end

#         if true
#             potential_next_sum = sum_so_far+calculate_trapezium_area(φ[curr_pos-dir],φ[curr_pos],slob.Δx)
#         else
#             potential_next_sum = sum_so_far+calculate_area(φ_interp,x_dense,slob.x[curr_pos],slob.x[curr_pos-dir])
#         end
#                                          # Record how much you could fulfill if you include the density you're on
#     end

#     positions_to_set_to_0 = (start_pos+dir):dir:(curr_pos-dir) # check which positions you completely used to fulfill the order and set them to 0
#     for i in  positions_to_set_to_0             # now loop over all the positions  
#                                                 # (not including where you started i.e. start_pos+dir or stopped i.e. curr_pos-dir)
#         D[slob_num].lob_densities[i,t-1] = 0.0  # set all these to 0
#         if slob.do_exp_dist_times
#             D[slob_num].lob_densities_L[i+1,t-1] = 0.0  # set all these to 0 #was -
#             D[slob_num].lob_densities_R[i-1,t-1] = 0.0  # set all these to 0 #was +
#         else
#             D[slob_num].lob_densities_L[i,t-1] = 0.0  # set all these to 0
#             D[slob_num].lob_densities_R[i,t-1] = 0.0  # set all these to 0
#         end

#     end


#     if do_trapezium
#         if curr_pos == start_pos + dir
#             farthest_zero = slob.x[start_pos] + dir * frac * slob.Δx + 
#                             dir * calculate_distance_to_get_volume(0,φ[curr_pos],slob.Δx*(1-frac),amount)

#             p = amount/calculate_trapezium_area(0,φ[curr_pos],(1-frac)*slob.Δx)
#         else
#             farthest_zero = slob.x[curr_pos-dir] +
#                             dir * calculate_distance_to_get_volume(φ[curr_pos-dir],φ[curr_pos],slob.Δx,amount-sum_so_far)

#             p = (amount-sum_so_far)/calculate_trapezium_area(φ[curr_pos-dir],φ[curr_pos],slob.Δx)
#         end
#     else
#         if curr_pos == start_pos + dir
#             farthest_zero = calculate_distance_to_get_volume_smooth(φ_interp,x_dense,amount,closest_zero)

#             p = amount/calculate_area(φ_interp,x_dense,slob.x[curr_pos],closest_zero)
#         else
#             farthest_zero = calculate_distance_to_get_volume_smooth(φ_interp,x_dense,amount-sum_so_far,slob.x[curr_pos-dir])

#             p = (amount-sum_so_far)/calculate_area(φ_interp,x_dense,slob.x[curr_pos],slob.x[curr_pos-dir])
#         end
#     end

#     D[slob_num].raw_price_paths[t-1] = (farthest_zero + closest_zero)/2.0

#     if do_smoothing
#         new_val = (1 - p) * φ[curr_pos]
#         D[slob_num].lob_densities[curr_pos,t-1] = new_val
#         if slob.do_exp_dist_times
#             D[slob_num].lob_densities_L[curr_pos+1,t-1] = new_val #was -
#             D[slob_num].lob_densities_R[curr_pos-1,t-1] = new_val #was +
#         else
#             D[slob_num].lob_densities_L[curr_pos,t-1] = new_val
#             D[slob_num].lob_densities_R[curr_pos,t-1] = new_val
#         end
#     end
# end

# +
# function my_interpolator(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64})::AbstractInterpolation
#     A = hcat(x,y)
#     return Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))))), x, 1:2)
# end

# +
# function auto_interpolator(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64},x_dense::AbstractRange;check_zeros::Bool=false)::Vector{Float64}
#     if check_zeros
#         if sum(abs.(y))<1e-13
#             return [0.0 for xi in x_dense]
#         end
#     end
#     itp = my_interpolator(x,y)
#     y_func = (t) -> itp(t,2)
#     return [y_func(xi) for xi in x_dense]
# end

# +
# function get_interpolated_intercept(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64},left_x_bound::Float64,right_x_bound::Float64)::Float64
#     itp = my_interpolator(x,y)
#     y_func = (t) -> itp(t,2)
#     return find_zero(y_func, (left_x_bound,right_x_bound), Bisection())
# end

# +
# using Interpolations, Plots, Roots

# # Lower and higher bound of interval
# a = 1.0
# b = 10.0
# # Interval definition
# x = a:1.0:b
# # This can be any sort of array data, as long as
# # length(x) == length(y)
# y = @. cos(x^2 / 9.0) # Function application by broadcasting
# # Interpolations
# itp_linear = linear_interpolation(x, y)
# itp_cubic = cubic_spline_interpolation(x, y)
# # Interpolation functions
# f_linear(x) = itp_linear(x)
# f_cubic(x) = itp_cubic(x)
# # Plots
# width, height = 1500, 800 # not strictly necessary
# x_new = a:0.1:b # smoother interval, necessary for cubic spline

# scatter(x, y, markersize=10,label="Data points")
# plot!(f_linear, x_new, w=3,label="Linear interpolation")
# plot!(f_cubic, x_new, linestyle=:dash, w=3, label="Cubic Spline interpolation")
# plot!(size = (width, height))
# plot!(legend = :bottomleft)

# +
# smoothed = auto_interpolator(x,y,x_new)
# get_interpolated_intercept(x,y,3.0,4.0)
