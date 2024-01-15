# -*- coding: utf-8 -*-
# +
function initial_conditions_numerical(D::Dict{Int64, DataPasser}, 
                                        t::Int64, slob_num::Int64, V₀::Float64=0.0)::Nothing
    slob¹ = D[slob_num].slob
    #slob² = D[2+(1-slob_num)].slob
    
    if (t == 1)
        t = t + 1
    end# ?????????????????????????????????????????????????/
    
    # eq 36:
    ud = (-V₀/(2.0*slob¹.Δx) + slob¹.D/(slob¹.Δx^2)) * ones(Float64, slob¹.M)  #upper diagonal
    md = ((-2.0*slob¹.D)/(slob¹.Δx^2) - slob¹.nu) * ones(Float64, slob¹.M+1)   #middle diagonal
    ld = (V₀/(2.0*slob¹.Δx) + slob¹.D/(slob¹.Δx^2)) * ones(Float64, slob¹.M)   #lower diagonal
    A = Tridiagonal(ld, md, ud)

    A[1,2] = 2*slob¹.D/(slob¹.Δx^2)
    A[end, end-1] = 2*slob¹.D/(slob¹.Δx^2)
    
    source   = source_function(   D,
                                    slob_num, t) 

    coupling = coupling_function( D,
                                    slob_num, t) 

    rl_push  = rl_push_function(  D,
                                    slob_num, t) 
    
    B = .-(source.+coupling.+rl_push)#B is s in (37)
    
    #temp
    if slob¹.source_term.do_source
        φ = A \ B
    else 
        φ = source .* 0
    end
    
    D[slob_num].lob_densities[:,t] = φ
    
    if slob¹.do_exp_dist_times
        D[slob_num].lob_densities_L[:,t] = change_side(φ,1 ,-1)
        D[slob_num].lob_densities_R[:,t] = change_side(φ,-1, 1)
    else
        D[slob_num].lob_densities_L[:,t] = φ
        D[slob_num].lob_densities_R[:,t] = φ
    end
    
    D[slob_num].sources[:,t]       = source
    D[slob_num].couplings[:,t]     = coupling
    D[slob_num].rl_pushes[:,t]     = rl_push
    D[slob_num].V[t]               = V₀
    
    #D[slob_num].RLBrains[t]        =  ?Already initialised correctly?
    #D[slob_num].RLViews[t]          = RLView([1.0,1.0],0.0)
    c = D[slob_num].RLParam.c
    len = length(c)
    D[slob_num].RLViews[t]         = RLView([1.0,1.0],0.0)
    D[slob_num].RLViews[t-1]         = RLView([1.0,1.0],0.0)
    D[slob_num].RLBrains[t]        =    ContinuousLearning.copy_brain(D[slob_num].RLParam.init_brain) #ContinuousLearning.make_brain(len)
    D[slob_num].RLBrains[t-1]        = ContinuousLearning.copy_brain(D[slob_num].RLParam.init_brain) #ContinuousLearning.make_brain(len)
    
    #give x such that A*x = B
    #return φ, source, coupling, rl_push, V₀
    return nothing
    
end

# +
function extract_mid_price_index_efficient(slob::SLOB,lob_density::Vector{Float64},previous_pos::Int64)::Int64
    if previous_pos < 0
        return -1
    end
    
    #l = slob.M+1
    l = length(lob_density)
    
    #    1  ...   _     _     _     _   previous_pos     _    _    _    ... l
    #    1  ...   _     _     _   left     right         _    _    _    ... l
    
    left_ = previous_pos - 1 
    right_ = previous_pos 
    
    if sign( lob_density[left_] ) * sign( lob_density[right_] ) < 0.0
        return right_
    end
    
    while (left_ > 1) && (right_ < l)
        
        if (sign( lob_density[left_] ) * sign( lob_density[left_ - 1] ) < 0.0)  
            return left_
        end
        
        if (sign( lob_density[right_] ) * sign( lob_density[right_ + 1] ) < 0.0) 
            return right_ + 1
        end
        
        if ((lob_density[left_] == 0.0) && (sign(lob_density[left_ - 1]) * sign(lob_density[left_ + 1]) < 0.0)) 
            return left_
        end
        
        if ((lob_density[right_] == 0.0) && (sign(lob_density[right_ - 1]) * sign(lob_density[right_ + 1]) < 0.0)) 
            return right_
        end
                
        
        left_  -= 1
        right_ += 1
        
    end 
        
        
#     # only do for cyclic boundary coditions
#     if (sign( lob_density[1] ) * sign( lob_density[l] ) < 0.0) ||
#         ((lob_density[1] == 0.0) && (sign(lob_density[l]) * sign(lob_density[2]) < 0.0))  
        
#         return 1
#     end
        
#     if ((lob_density[l] == 0.0) && (sign(lob_density[l-1]) * sign(lob_density[1]) < 0.0))
        
#         return l
        
#     end
        
    
    
    if left_ == 1           #if we hit the left wall

        curr_right_ = right_
        right_ = l
            
        while (right_ >= curr_right_)

            right_ -= 1
            
            if (sign( lob_density[right_] ) * sign( lob_density[right_ + 1] ) < 0.0) 
                return right_ + 1
            end

            if ((lob_density[right_] == 0.0) && (sign(lob_density[right_ - 1]) * sign(lob_density[right_ + 1]) < 0.0))  
                return right_
            end
                
            

        end


    else# right_ == l+1           we must have hit the right wall

        curr_left_ = left_
        left_ = 0
        while (left_ <= curr_left_)

            left_ += 1

            if (sign( lob_density[left_] ) * sign( lob_density[left_ - 1] ) < 0.0)  
                return left_
            end
            
            if ((lob_density[left_] == 0.0) && (sign(lob_density[left_ - 1]) * sign(lob_density[left_ + 1]) < 0.0))  
                return left_
            end

        end
    end
   
    return -1
end
# -

function extract_mid_price_index(slob::SLOB,lob_density::Vector{Float64},previous_pos::Int64)::Int64
    return extract_mid_price_index_efficient(slob,lob_density,previous_pos)
end

function map_back(index,size)
    #return mod(index-1,size)+1
    return index
end

function my_interpolator(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64})::AbstractInterpolation
    A = hcat(x,y)
    return Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))))), x, 1:2)
end

function auto_interpolator(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64},x_dense::AbstractRange;check_zeros::Bool=false)::Vector{Float64}
    if check_zeros
        if sum(abs.(y))<1e-13
            return [0.0 for xi in x_dense]
        end
    end
    itp = my_interpolator(x,y)
    y_func = (t) -> itp(t,2)
    return [y_func(xi) for xi in x_dense]
end

function get_interpolated_intercept(x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},y::Vector{Float64},left_x_bound::Float64,right_x_bound::Float64)::Float64
    itp = my_interpolator(x,y)
    y_func = (t) -> itp(t,2)
    return find_zero(y_func, (left_x_bound,right_x_bound), Bisection())
end

function SibuyaKernelDP(n::Int64,slob::SLOB)::Float64
    return slob.SK_DP[n]
end

function calculate_next_step_original(φ::Vector{Float64}, slob::SLOB, P::Float64, P⁺::Float64, P⁻::Float64,net_source::Vector{Float64})::Vector{Float64} #NB, pass in only most recent time
    #φ = D[slob_num].lob_densities[:,t_current-1]
    
    ###### LITERALLY Michael's code #####################################
            
    φ₋₁ = φ[1]
    φ₊₁ = φ[end]
    φ_next = zeros(Float64, size(φ,1))

    φ_next[1] = P⁺ * φ₋₁ + P⁻ * φ[2] + P * φ[1] -
        slob.nu * slob.Δt * φ[1] + slob.Δt * net_source[1]

    φ_next[end] = P⁻ * φ₊₁ + P⁺ * φ[end-1] + P * φ[end] -
        slob.nu * slob.Δt * φ[end] + slob.Δt * net_source[end]

    φ_next[2:end-1] = P⁺ * φ[1:end-2] + P⁻ * φ[3:end] + P * φ[2:end-1] -
        slob.nu * slob.Δt * φ[2:end-1] + slob.Δt * net_source[2:end-1]
    
    #####################################################################
    
    return φ_next
end

# +
function calculate_next_step_no_exp(φ::Matrix{Float64},φ_L::Matrix{Float64},φ_R::Matrix{Float64}, slob::SLOB, P::Float64, P⁺::Float64, P⁻::Float64,net_source,
                                t::Int64)#::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    myend = size(φ,1)
    φ_next = zeros(Float64,myend)
    middle = 2:(myend-1)    #all indices excluding the first and last
    
    Pm = P - 1                                   # becomes ( 1 - r ) - 1 = -r                               
    
    removal_scale = exp( - slob.nu * slob.Δt)
    
    test_val = 0.0
    for tₘ in max(1,t-slob.cut_off):(t-1)
      
        front = SibuyaKernelDP(t-tₘ,slob)*exp(- ((t-tₘ+slob.shift) * slob.Δt) * slob.nu) #shift is usually -1, this option is for legacy reasons
        
        φ_next[1] += front * 
            (  P⁺ * φ[1          ,tₘ] +   #1      -1    (but not because boundary)
               P⁻ * φ[2          ,tₘ] +   #1      +1
               Pm * φ[1          ,tₘ]  )  #1      +0 

        φ_next[end] += front * 
            (  P⁺ * φ[end-1      ,tₘ] +   #end    -1  
               P⁻ * φ[end        ,tₘ] +   #end    +1    (but not because boundary)
               Pm * φ[end        ,tₘ]  )  #end    +0

        φ_next[middle] +=  front * 
            (  P⁺ * φ[middle.-1  ,tₘ] +   #middle -1
               P⁻ * φ[middle.+1  ,tₘ] +   #middle +1 
               Pm * φ[middle     ,tₘ]  )  #middle +0 
        
        # the above is a more efficient version of 
        # φ_next += front * 
        #    ( P⁺ * change_side(φ[:,tₘ],1,-1) + 
        #      P⁻ * change_side(φ[:,tₘ],-1,1) + 
        #      Pm * φ[:,tₘ] )
            
    end
    
    φ_next += removal_scale * φ[:,t-1] + slob.Δt * net_source
        
    return (φ_next,φ_next,φ_next,-1)
    
end

# +
function calculate_next_step_exp(φ::Matrix{Float64},φ_L::Matrix{Float64},φ_R::Matrix{Float64}, slob::SLOB, P::Float64, P⁺::Float64, P⁻::Float64,net_source,
                                t::Int64)#::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    myend = size(φ,1)
    φ_next = zeros(Float64,myend)
    φ_L_next = zeros(Float64,myend)
    φ_R_next = zeros(Float64,myend)

    Pm = P - 1                                   # becomes ( 1 - r ) - 1 = -r                               
    removal_scale = exp( - slob.nu * slob.Δts[t-1])

    #test_val = 0.0
    for tₘ in max(1,t-slob.cut_off):(t-1)
        
        #front = SibuyaKernelDP(t-tₘ,slob)*exp(- (slob.Δts_cum[t+slob.shift] -  slob.Δts_cum[tₘ]) * slob.nu) #shift is usually -1, this option is for legacy reasons
        front = SibuyaKernelDP(t-tₘ,slob)*exp(- sum(slob.Δts[tₘ:t+slob.shift-1]) * slob.nu) #shift is usually -1, this option is for legacy reasons
        ##################### NB, could need to be !!!!!!!!!!!!!!!!
            ##front = SibuyaKernelDP(t-tₘ,slob)*exp(- sum(slob.Δts[(tₘ+1):t+slob.shift]) * slob.nu) #shift is usually -1, this option is for legacy reasons

        φ_next += front * 
            (  P⁺ * φ_L[:        ,tₘ] + 
               P⁻ * φ_R[:        ,tₘ] + 
               Pm *   φ[:        ,tₘ]  ) 
    end
    
    
    φ_next += removal_scale * φ[:,t-1] + slob.Δts[t-1] * net_source
    
    φ_next, φ_L_next, φ_R_next = dr_g_way(φ_next,slob,t)
    #φ_next, φ_L_next, φ_R_next = linear_way(φ_next,slob,t)
    #φ_next, φ_L_next, φ_R_next = dr_g_way_modified(φ_next,slob,t)

    return (φ_next, φ_L_next, φ_R_next, -1)
      
end

# +
function dr_g_way(φ_next,slob,t)
    # DR G WAY but inefficient
    
    myend = size(φ_next,1)
    k = floor(Int64, slob.Δxs[t-1]/slob.Δx) # 1 if doing uniform distribution
    prop = (slob.Δxs[t-1] - k * slob.Δx)/(2 * slob.Δx) # 0 if doing uniform distribution

    φ_next_ext = change_side(φ_next,k+1,k+1) #insert k+1 repeats onto each side
    middle_ext = (1:myend) .+ (k+1) #select the middle region without repeats

    middle_right = middle_ext .+ k #select the right k repeats
    φ_R_next = φ_next_ext[middle_right] .+ prop * (φ_next_ext[middle_right.+1] .- φ_next_ext[middle_right.-1])

    middle_left = middle_ext .- k #select the left k repeats
    φ_L_next = φ_next_ext[middle_left] .- prop * (φ_next_ext[middle_left.+1] .- φ_next_ext[middle_left.-1])

    return (φ_next, φ_L_next, φ_R_next)
    
end

# +
function linear_way(φ_next,slob,t)
    myend = size(φ_next,1)
    k = floor(Int64, slob.Δxs[t-1]/slob.Δx) # 1 if doing uniform distribution
    prop = (slob.Δxs[t-1] - k * slob.Δx)/(slob.Δx) # 0 if doing uniform distribution

    φ_next_ext = change_side(φ_next,k+1,k+1) #insert k+1 repeats onto each side
    middle_ext = (1:myend) .+ (k+1) #select the middle region without repeats

    middle_right = middle_ext .+ k #select the right k repeats
    φ_R_next = φ_next_ext[middle_right] .+ prop * (φ_next_ext[middle_right.+1] .- φ_next_ext[middle_right])

    middle_left = middle_ext .- k #select the left k repeats
    φ_L_next = φ_next_ext[middle_left] .+ prop * (φ_next_ext[middle_left.-1] .- φ_next_ext[middle_left])

    return (φ_next, φ_L_next, φ_R_next)
    
end

# +
function dr_g_way_modified(φ_next,slob,t)
    myend = size(φ_next,1)
    k = floor(Int64, slob.Δxs[t-1]/slob.Δx) # 1 if doing uniform distribution
    prop = (slob.Δxs[t-1] - k * slob.Δx)/(2 * slob.Δx) # 0 if doing uniform distribution

    φ_next_ext = change_side(φ_next,k+1,k+1) #insert k+1 repeats onto each side
    middle_ext = (1:myend) .+ (k+1) #select the middle region without repeats

    middle_right = middle_ext .+ k #select the right k repeats
    
    diff_ = φ_next_ext[middle_right.+1] .- φ_next_ext[middle_right.-1]
    avg1_ = (φ_next_ext[middle_right.+1] .+ φ_next_ext[middle_right.-1])./2
    
    sum_ = ((2*prop * avg1_) .+ ((1-2*prop)*φ_next_ext[middle_right]))
    
    φ_R_next = sum_ .+ prop * diff_

    middle_left = middle_ext .- k #select the left k repeats
    diff_ = φ_next_ext[middle_left.+1] .- φ_next_ext[middle_left.-1]
    avg1_ = (φ_next_ext[middle_left.+1] .+ φ_next_ext[middle_left.-1])./2
    sum_ = ((prop * avg1_) .+ ((1-prop)*φ_next_ext[middle_left]))
    φ_L_next = sum_ .- prop * diff_

    return (φ_next, φ_L_next, φ_R_next)
    
end

# +
function intra_time_period_simulate_fractional( D::Dict{Int64, DataPasser},
                                                t::Int64, t_current::Int64, slob_num::Int64)::Nothing 
    slob = D[slob_num].slob
    
    special_function(             D,
                                    slob_num, t) 
    
    P, P⁺, P⁻, V_t  = randomness_function( D, 
                                    slob_num, t) 

    source   = source_function(             D,
                                    slob_num, t) 

    coupling = coupling_function(           D,
                                    slob_num, t) 

    rl_push  = rl_push_function(            D,
                                    slob_num, t) 

    net_source = source .+ coupling .+ rl_push
    
    #if slob.michaels_way
        ###### LITERALLY Michael's code #####################################
    #    φ_next = calculate_next_step_original(D[slob_num].lob_densities[:,t_current-1], slob, P, P⁺, P⁻, net_source)
        
    
    if slob.do_exp_dist_times
        (φ_next,φ_L_next,φ_R_next,test_val) = calculate_next_step_exp(     D[slob_num].lob_densities, D[slob_num].lob_densities_L, D[slob_num].lob_densities_R, 
                                                                  slob, P, P⁺, P⁻, net_source, t_current) 
    else
        (φ_next,φ_L_next,φ_R_next,test_val) = calculate_next_step_no_exp(  D[slob_num].lob_densities, D[slob_num].lob_densities_L, D[slob_num].lob_densities_R, 
                                                                  slob, P, P⁺, P⁻, net_source, t_current) 
    end
    
    
    #if 9 <= t <= 11
    #print("Time is ",t,
    #      " and max is ",#maximum(abs.(φ_next_temp.-φ_next)),
    #                     #maximum(abs.(vcat(φ_next_temp[1],φ_next_temp[1:end-1]).-φ_L_next)),
    #    
    #      " and test val is ", test_val ,"\n")
    #end

        
    # store all those intermediate values
 
    D[slob_num].lob_densities[:,t_current] = φ_next
    D[slob_num].lob_densities_L[:,t_current] = φ_L_next 
    D[slob_num].lob_densities_R[:,t_current] = φ_R_next 
    D[slob_num].sources[:,t_current]       = source
    D[slob_num].couplings[:,t_current]     = coupling
    D[slob_num].rl_pushes[:,t_current]     = rl_push
    D[slob_num].Ps[t]              = P
    D[slob_num].P⁺s[t]             = P⁺
    D[slob_num].P⁻s[t]             = P⁻
    D[slob_num].V[t]               = V_t
    
    #special function handles the below
    #D[slob_num].RLBrains[t]         
    #D[slob_num].RLViews[t]         
    
    return nothing

end
# -


function move_densities(D::Dict{Int64, DataPasser};how_many=1)::Nothing
    for time in 2:1:(1+how_many)
        for slob_num in 1:length(D)
            D[slob_num].lob_densities[:,time-1] = D[slob_num].lob_densities[:,time]
            D[slob_num].lob_densities_L[:,time-1] = D[slob_num].lob_densities_L[:,time]
            D[slob_num].lob_densities_R[:,time-1] = D[slob_num].lob_densities_R[:,time]
            D[slob_num].sources[:,time-1]       = D[slob_num].sources[:,time]
            D[slob_num].couplings[:,time-1]     = D[slob_num].couplings[:,time]
            #D[slob_num].rl_pushes[:,time-1]     = D[slob_num].rl_pushes[:,time]
            
            D[slob_num].RLBrains[time-1] = D[slob_num].RLBrains[time]
            D[slob_num].RLViews[time-1]  = D[slob_num].RLViews[time]
        end
    end
    
    return nothing
end

# +
function dtrw_solver_fractional(D::Dict{Int64, DataPasser},p,progress)::Int64 #handles change over time logic
    broke_point = -1
    slob = D[1].slob
    
    x_range_original = slob.x[:] #will not work for different starting ranges across x for different slobs
    
    time_steps = slob.N#to_simulation_time(slob.T, slob.Δt) 
    
    t = 1 #initial conditions take current t to read values

    for slob_num in 1:length(D)
        curr_slob = D[slob_num].slob
        D[slob_num].raw_price_paths[1] = curr_slob.p₀
        D[slob_num].obs_price_paths[1] = curr_slob.p₀
        initial_conditions_numerical( D, t, slob_num) # writes to position 2. Ugh. Fix later. So need to move back first
    end
    
    t = 2
    
    not_broke = reduce(&,[(D[l].raw_price_paths[t-1]!=-1 || !D[l].slob.source_term.do_source) for l in 1:length(D)],init=true) #only true when non of the most recent raw price values are -1
    
    if slob.store_past_densities
        move_back = (D) -> nothing
        t_current = identity
    else
        move_back = (D) -> move_densities(D;how_many = length(slob.SK_DP)) # moves the information in time step 2 to time step 1
        t_current = (t) -> length(slob.SK_DP)+1#2
    end
    
    while (t <= time_steps) && (not_broke)
        
        move_back(D)
    
        for slob_num in 1:length(D)
            intra_time_period_simulate_fractional( D, t, t_current(t), slob_num ) 
            
            
            D[slob_num].raw_price_paths[t] = 
                                    extract_mid_price(  D[slob_num].slob, 
                                                        D[slob_num].lob_densities[:,t_current(t)],
                                                        #price_to_index(D[slob_num].raw_price_paths[t-1]+D[slob_num].slob.price_shift,slob.Δx,slob.p₀-slob.L/2) )
                                                        price_to_index(D[slob_num].raw_price_paths[t-1],slob.Δx,slob.x[1]) )
            
            D[slob_num].x_shifts[t] = D[slob_num].x_shifts[t-1] + D[slob_num].slob.price_shift
            D[slob_num].slob.price_shift = 0.0
            
            if (progress) && (t%10==0)
                next!(p)
            end
        end
        
        t += 1 
        not_broke = reduce(&,[(D[l].raw_price_paths[t-1]!=-1 || !D[l].slob.source_term.do_source) for l in 1:length(D)],init=true)
    end
    
    #check if broken
    if !(not_broke)
        print("BE")
        broke_point = t-1
    end
    
    for slob_num in 1:length(D)
        D[slob_num].obs_price_paths = 
                    sample_mid_price_path(D[slob_num].slob, D[slob_num].raw_price_paths) 
        
        D[slob_num].slob.x = x_range_original[:]
    end
    
    return broke_point
    
end
# -
function change_side(φ,left_no,right_no)
    if      (left_no <  0) && (right_no <  0)
        return φ[1+(-left_no):end-(-right_no)]
    elseif (left_no >= 0) && (right_no <  0)
        return vcat(repeat([φ[1]], left_no) , φ[1:end-(-right_no)])
    elseif (left_no <  0) && (right_no >= 0)
        return vcat(φ[1+(-left_no):end],repeat( [φ[end]] ,right_no))
    else 
        return vcat( repeat( [φ[1]], left_no) , φ , repeat( [φ[end]] ,right_no) )
    end
end

# # Junk

# +
function extract_mid_price(slob::SLOB, lob_density::Vector{Float64}, previous_pos::Int64;)::Float64
    
    mid_price_ind = extract_mid_price_index(slob,lob_density,previous_pos)
    l = slob.M+1
    
    if (mid_price_ind == -1) || (mid_price_ind == 1)
        return -1
    end 
    
    left  = map_back(mid_price_ind-1,l)
    right = map_back(mid_price_ind  ,l)
    
    #change = mid_price_ind - previous_pos
    
    
    # if mid_price_ind == 1
    #     left = l
    #     right = 1
    # else
    #     left = mid_price_ind - 1
    #     right = mid_price_ind
    # end
    
    width = 7
    
#     if abs(change) > 0.5 * slob.M
#         if sign(change) > 0
#             slob.price_shift = -slob.L
#         else 
#             slob.price_shift = +slob.L
#         end
        
#         slob.x_range = (slob.x_range[1] + slob.price_shift) : slob.L/slob.M : (slob.x_range[end] + slob.price_shift)
#         slob.x = collect(Float64, slob.x_range)
#     end
         
    # in the below we have that the actual midprice lies between left=(mid_price_index - 1) and right=(mid_price_index)
    if (!slob.do_interp) || (right < width+1) || (left > slob.M - width + 2)
        y1 = lob_density[left] #y value (density) to the left
        y2 = lob_density[right]   #y value (density) to the right
        #x1 = slob.x[left]      #x value value to the left
        x2 = slob.x[right]      #x value value to the right
        
        
        #mid_price = (-y1 * slob.Δx)/(y2 - y1) + x1 
        mid_price = (-y2 * slob.Δx)/(y2 - y1) + x2 
        
        #solution to assuming straight approximation between 
                                                   #left and right point (did the math) (see page 20)
        
        
    else
        if slob.cyclic_boundary 
                print("Shouldn't have gotten here")
        end
        
        x2 = slob.x[mid_price_ind]   #x value value to the right
        
        if lob_density[mid_price_ind]!=0.0
            x1 = slob.x[mid_price_ind-1] #x value value to the left

            lower = mid_price_ind-width
            upper = mid_price_ind+width-1
            #print(upper," ")
            x_range = slob.x[lower]:slob.Δx:slob.x[upper]

            mid_price = get_interpolated_intercept(     x_range,
                                      lob_density[lower:upper],x1,x2)
        else
            mid_price = x2
        end
    end
    
    return mid_price
end

