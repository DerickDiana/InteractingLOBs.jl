# -*- coding: utf-8 -*-
mutable struct SLOB
    num_paths::Int64 #How many paths to simulate
    T::Int64    # How many time periods it runs for
    
    p₀::Float64 # The initial mid price
    M::Int64    # Number of discrete price points in addition to p₀ (the central point)
    N::Int64
    Δt::Float64 
    Δx::Float64
    
    L::Float64  # The length of the price dimension in the lattice
    D::Float64  # The diffusion constant (determines Δt)
    nu::Float64 # Cancellation rate
    α::Float64  # Waiting time in terms of Exp(α)
    
    source_term::SourceTerm # Source term s(x,t)(μ,λ,p¹,p²) 
    coupling_term::CouplingTerm # Interaction term c(x,t)(μ,λ,p¹,p²)
    rl_push_term::RLPushTerm
    randomness_term::RandomnessTerm
    
    x::Array{Float64, 1}    #
    x_range::AbstractRange
    Δxs::Array{Float64, 1} # The grid spacing which is calculated as Δx=L/M
    Δts::Array{Float64, 1} # The temporal grid spacing calculated as Δt=Δx*Δx/(2D)
    Δts_cum::Array{Float64, 1} 
    #Δts_diff::Matrix{Float64}
    
    γ::Float64 #Degree of fractional diffusion
    zero_vec::Array{Float64,1}
    
    SK_DP::Array{Float64,1}
    cut_off::Int64
    
    shift::Int64
    old_way::Bool
    michaels_way::Bool
    scale::Float64
    kernel_cut_off::Float64
    store_past_densities::Bool
    
    cyclic_boundary::Bool
    price_shift::Float64
    
    do_interp::Bool
    do_exp_dist_times::Bool
    do_test::Bool
    do_debug::Bool
end


"""
    SLOB(<many arguments>; <keyword arguments>)

The above is a constructor for a Struct of type SLOB. When called, one must supply all of the following arguments:

# Main arguments
- `num_path::Int64`:
- `T::Int64`:
- `p₀::Float64`:
- `Δt::Float64`:
- `L::Float64`:
- `D::Float64`:
- `nu::Float64`:
- `α::Float64`:
- `γ::Float64`:
- `source_term::SourceTerm`:
- `coupling_term::CouplingTerm`:
- `rl_push_term::RLPushTerm`:
- `randomness_term::RandomnessTerm`:

# Keyword arguments
- `shift::Int64=-1`: See paper, whether to shift the integration of past information back by -1 (can also be set to 0)
- `old_way::Bool=false`: Use Michael's code exactly
- `michaels_way::Bool=false`: Use Michael's code exactly
- `scale::Float64=1.0`: scale the conversion from `Δx` to `Δt` (not used anymore)
- `kernel_cut_off::Float64=0.01`: how small must contributing terms in the kernel calculation be before you stop using them
- `store_past_densities::Bool=true`: Whether to store all the densities used for display in `plot_density_visual`. Set to false 
                                    to save memory when doing long calculations.
- `do_interp::Bool=false`: Use smooth interpolation to find the intercept
- `do_cyclic_boundary::Bool=false`: Use cyclic boundary conditions
- `do_exp_dist_times::Bool=false`: Generate the set of `Δts` from an exponential distribution
- `do_test::Bool=false`: Use the exponential algorithm code but make the generated `Δts` be uniform across time
- `Δx::Float64=-1`: Specify the background grid `Δx`. If left as -1, calculated from given `Δt` using `calculate_Δx_from_Δt` function.
- `do_debug::Float64=false`: Functionality changes. Used for inserting debugging code on the fly.
- `seed::Int64=-1`: What seed to pass to `InteractOrderBooks` function. First modified by the function `obtain_seed`
- `beginning_shift_frac::Float64=0.0`: When the initial simulation is set up, the intercept will be on a grid point by default. One may include 
                                     a non-zero value for this parameter to shift the starting point of the intercept by a fraction of `Δx`. That is
                                     the initial intercept will end up at a position: `p₀ + beginning_shift_frac * Δx`

"""
function SLOB(num_paths::Int64, T::Int64, p₀::Float64,
    Δt::Real, 
    L::Real, D::Float64, nu::Float64, 
    α::Float64, γ::Float64, 
    source_term::SourceTerm, coupling_term::CouplingTerm, rl_push_term::RLPushTerm, randomness_term::RandomnessTerm;
    shift=-1,old_way=false,michaels_way=false,scale=1.0,kernel_cut_off=0.00001,store_past_densities=true,
    do_interp=false,do_cyclic_boundary=false, do_exp_dist_times=false, do_test=false, Δx=-1, do_debug=false, seed=-1, beginning_shift_frac=0.0) 
    #translates convinient order into order expected by object itself
    
    
    r = randomness_term.r
    
    if Δx==-1
        Δx = calculate_Δx_from_Δt(Δt,γ,D,r)
    end
    
    seed = obtain_seed(seed)
    
    x₀ = p₀ - 0.5*L
    x_m = p₀ + 0.5*L
    
    price_shift = 0.0
    
    x_range = x₀:Δx:x_m #implies a length M+1 vector
    x = collect(Float64, x_range) #creates an array of the entries. So returns set of x points
    zero_vec = 0.0 .* x[:]
    M = length(x)-1
    #print("Got here with T = ",T," and Δt = ",Δt,"\n")
    N = to_simulation_time(T,Δt)
    # NB, the fact that you decided the simulation would run for N steps before doing the kernel could be an issue. May need to put * below, above this

    SK_DP = calculate_modified_sibuya_kernel(γ,nu,T,Δt;shift=shift,kernel_cut_off=kernel_cut_off)
    cut_off = length(SK_DP)
    
    
    # *
    if do_exp_dist_times && !do_test
       
        Δts = generate_Δts_exp(T,Δt;seed=seed) 
        
        Δxs = (Δt_ -> calculate_Δx_from_Δt(Δt_,γ,D,r)).(Δts)
        
        N = length(Δts)
        
        Δts_cum = vcat(0.0,cumsum(Δts))
    else
        Δts = repeat([Δt],N)
        
        Δxs = repeat([Δx],N)
        
        Δts_cum = vcat(0:N) .* Δt
    end
    
    #Δts_diff = [[Δts_cum[j] - Δts_cum[i] for i in 1:N] for j in 1:N] #get time at i - time at j by doing [i][j] (larger number first)
    p₀ += beginning_shift_frac * Δx
    
    return SLOB(num_paths, T, p₀, 
        M, N, Δt, Δx,
        L, D, nu, α, 
        source_term, coupling_term, rl_push_term, randomness_term, 
        x, x_range, Δxs, Δts, Δts_cum, 
        γ, zero_vec, SK_DP,
        cut_off,shift,old_way,michaels_way,scale,kernel_cut_off,store_past_densities,
        do_cyclic_boundary,price_shift,do_interp,do_exp_dist_times,do_test,do_debug)
end

function generate_Δts_exp(T,Δt;seed=-1)
    
    #if seed == -1
    #    seed = floor(Int64,rand() * 1e10)
    #end
    
    seed = obtain_seed(seed)
    
    target_number_of_steps = to_simulation_time(T,Δt) # calculate target number of steps assuming uniform distribution of Δts
    
    Random.seed!(seed)
    Δts = rand(Exponential(Δt), target_number_of_steps)

    multiplier = 1.0
    while sum(Δts) < T
        multiplier = multiplier +  1.0
        
        Random.seed!(seed)
        Δts = rand(      Exponential(Δt),     floor(Int64, target_number_of_steps * multiplier)      )
    end

    temp_cum_Δts = vcat(0.0,cumsum(Δts))
    last_pos = to_simulation_time(T,Δt;do_exp=true,cum_Δts = temp_cum_Δts ) - 2 ###
    
    # EXPLANATION FOR -2. 
    # suppose we want to include Δts up to just before they sum to 1.0 or more. 
    # Let         Δts = 0.7           0.2           0.4               0.2
    
    # if cum_Δts does not start with 0.0:
    #Δts pos:            1            (2)             3                 4
    #Δts:               0.7           0.2           0.4               0.2
    #cum_Δts pos:        1             2            (3)                 4
    #cum_Δts:           0.7           0.9           1.3               1.5
    #                                  X        first_exceed_1 
    # first exceedance is (3), but we want (2), so we want Δts[1:first_exceed_1-1] (the brackets show the correct answers)

    # if cum_vals does start with 0.0:
    #Δts pos:                          1            (2)             3              4
    #Δts:                             0.7           0.2           0.4            0.2
    #cum_Δts pos:        1             2             3            (4)              5
    #cum_Δts:           0.0           0.7           0.9           1.3            1.5
    #                                                X        first_exceed_1 
    # first exceedance is (4), but we want (2), so we want Δts[1:first_exceed_1-2]  (the brackets show the correct answers)
    
    
    
    Δts = Δts[1:last_pos] # take only enough needed to fall short of passing the time needed

    Δts = vcat(Δts,[T-sum(Δts)+1e-13]) #insert final time to land exactly on the total time asked for
    return Δts
end

# +
function calculate_Δt_from_Δx(Δx,γ,D,r)
    Δt = (r * (Δx^2) / (2.0 * D))^(1/γ)
    return Δt
end

function calculate_Δx_from_Δt(Δt,γ,D,r)
    Δx = Δt^(γ/2)*(2.0 * D / r)^0.5
    return Δx
end
# -

function calculate_modified_sibuya_kernel(γ,nu,T,Δt;shift=-1,kernel_cut_off=0.01)
    function partially_applied_Sibuya(n)
         return SibuyaKernelModified(n,γ,nu,Δt)
    end

    l = to_simulation_time(T,Δt)
    SK_DP = zeros(Float64,l)

    cut_off = 0

    # basic loop
    next_sk = γ #first value of kernel is just gamma
    SK_DP[cut_off+1] = next_sk#*exp(-(cut_off+1+shift)*nu*Δt)                                   
    cut_off += 1                                    

    # special case for cut_off = 2
    next_sk = (-1+γ)*(1-(2-γ)/(cut_off+1)) #evaluates to (-1+γ)*(γ/2)
    SK_DP[cut_off+1] = next_sk#*exp(-(cut_off+1+shift)*nu*Δt)                                   

    # check if latest meets condition to be included. Only then do we increase the cutoff and calculate again
    while (cut_off<(l-1)) && (SK_DP[cut_off+1]<=-kernel_cut_off)            
        cut_off += 1

        next_sk = next_sk * (1-(2-γ)/(cut_off+1)) 
        SK_DP[cut_off+1] = next_sk #* exp(-(cut_off+1+shift)*nu*Δt)                    
    end

    SK_DP = SK_DP[1:cut_off] #shorten the array to whatever cutoff was found
    
    return SK_DP
end

function price_to_index(price::Float64,Δx::Float64,base_x::Float64)::Int64
    #return floor(Int,(price-base_x)/Δx)
    return floor(Int,(price-base_x)/Δx)
end

function index_to_price(index::Int64, Δx::Float64,base_x::Float64)::Float64 #should never be needed
      return base_x + index * Δx
end

# +
#function to_simulation_time(real_time, Δt::Float64) #from real time 
#    return simulation_time = floor(Int,real_time/ Δt + 1e-13)+1
#end
# -


function clear_double_dict(D)
    names = fieldnames(typeof(D))

    for (key1,value1) in D
        outer_dict = D[key1]
        for (key2,value2) in outer_dict
            inner_dict = outer_dict[key2]
            for name in names
               getfield(inner_dict,name) = nothing
            end
        end
    end
end

function get_sample_inds_by_sampling_at_integer_real_times(slob::SLOB) #sample the simulation at integer times in real time
    #if !slob.do_exp_dist_times
        partially_applied_to_sim_time = (real_times) -> to_simulation_time(real_times, slob.Δt; 
                                                        do_exp = slob.do_exp_dist_times, cum_Δts = slob.Δts_cum)
        my_real_times = vcat(0:(slob.T-1))
    #else
    #    partially_applied_to_sim_time = (real_times) -> to_simulation_time(real_times, slob.Δt; 
    #                                                    do_exp = slob.do_exp_dist_times, cum_Δts = slob.Δts_cum)
    #    my_real_times = vcat(0:slob.T)
    #end
    
    sample_inds = partially_applied_to_sim_time.(my_real_times)
    
    return sample_inds
end

function sample_mid_price_path(slob::SLOB, price_path)
    sample_inds = get_sample_inds_by_sampling_at_integer_real_times(slob)
    mid_prices = price_path[sample_inds[1:end-1]]
    return mid_prices
end

function InteractOrderBooks(slobs::Array{SLOB,1}, RLParams::Vector{RLParam}, seed::Int=-1, progress = false; return_break_points=false, progress_bar = -1, finish_progress_bar = true)#same but returns more stuff
    #handles change over paths logic
    #print(slobs[1].scale)
    # WHERE IS RLBRAIN?
    
    slob = slobs[1]
    
    # do some logging IDK
    #logger = FileLogger(Dict(Logging.Info => "info.log", Logging.Error => "error.log"), append=false)
    #global oldglobal = global_logger(logger)
    
    # generate a set of seeds
    if seed == -1
        seed = floor(Int64,rand() * 1e10)
    end
    
    #@info string("Seed used is: ",seed)
    
    seeds = Int.(rand(MersenneTwister(seed), UInt32, slob.num_paths)) #generate one seed for each path
    
    num_time_steps = slob.N  #to_simulation_time(slob.T, slob.Δt) #how many num_time steps in the base simulation? i.e.Raw num_time indices
    
    total_steps = floor(Int64, slob.num_paths*length(slobs)*num_time_steps/10)
    #@info string("Total steps is ",total_steps*1000,"\n")
    if progress_bar == -1
        progress_bar = Progress(total_steps,dt=0.1)
    end
    
    
    if slob.store_past_densities
        num_time_steps_for_densities = num_time_steps 
    else
        num_time_steps_for_densities = length(slob.SK_DP)
    end
    
    # the below dimensions are ordered according to the frequency with which they appear
    
    
    outer_dict = Dict{Int64,Dict{Int64,DataPasser}}()
    for path_num in 1:slob.num_paths
        inner_dict = Dict{Int64,DataPasser}()
        for slob_num in 1:length(slobs)
            ### reserve memory
            lob_densities =    zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)
            lob_densities_L =  zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)
            lob_densities_R =  zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)
            sources =          zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)
            couplings =        zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)
            rl_pushes =        zeros(Float64, slob.M+1, num_time_steps_for_densities + 1)

            raw_price_paths =  ones(Float64,            num_time_steps               + 1)
            obs_price_paths =  ones(Float64,            slob.T                       + 1)

            P⁺s =              ones(Float64,            num_time_steps               + 1)
            P⁻s =              ones(Float64,            num_time_steps               + 1)
            Ps =               ones(Float64,            num_time_steps               + 1)
            V =                ones(Float64,            num_time_steps               + 1)
            x_shifts =          zeros(Float64,            num_time_steps               + 1)
            
            RLBrains = my_repeat(RLBrain,                  num_time_steps_for_densities + 1)
            RLViews  = my_repeat(RLView,                   num_time_steps_for_densities + 1)
            

            my_struct = DataPasser(slobs[slob_num], lob_densities, lob_densities_L, lob_densities_R, sources, couplings, rl_pushes,
                                    raw_price_paths, obs_price_paths,
                                    P⁺s, P⁻s, Ps, V, x_shifts, RLBrains , RLParams[slob_num], RLViews,[0.7])
            ### end of reserve memory
            
            inner_dict[slob_num] = my_struct
        end
        
        outer_dict[path_num] = inner_dict
        
    end
    
    broke_points =     ones(Float64,             slob.num_paths)
    
    recalc = slob.α > 0.0 
    
    counter = 0
    
    for path_num = 1:slob.num_paths
    #Threads.@threads for path_num in 1:slob.num_paths  
    #Threads.@threads for (key,dict) in outer_dict  #paralellize over elements of dictionary? "dict" is the value in (key,value) and is an entire dictionary but for just one path
        
        Random.seed!(seeds[path_num])
        
        broke_point = dtrw_solver_fractional(outer_dict[path_num],progress_bar,progress)
        
        broke_points[path_num] = broke_point
       
        counter += 1
    end
    
    if (progress && finish_progress_bar)
        finish!(progress_bar)
    end

    if !return_break_points
        return   outer_dict
    else
        return (outer_dict,broke_points)
    end
end

function InteractOrderBooks(slobs::Array{SLOB,1}, seed::Int=-1, progress = false; return_break_points=false, progress_bar = -1, finish_progress_bar = true)#same but returns more stuff
    return InteractOrderBooks(slobs,my_repeat(RLParam,length(slobs)),seed,progress;return_break_points=return_break_points,progress_bar=progress_bar,finish_progress_bar=finish_progress_bar)
end

# +
# function to_real_time(simulation_time::Int64, Δt::Float64; do_exp = false, cum_Δts = []) # from simulation time
#     if do_exp
        
#         # correct
#         return real_time = floor(Int,cum_Δts[simulation_time])
#         ### return real = cum_Δts[simulation_time] ###
#     else
#         # correct
#         return real_time = floor(Int,(simulation_time-1) * Δt)
#     end
# end

# +
# function to_simulation_time(real_time, Δt::Float64;do_exp=false,cum_Δts=[]) #from real time 
#     if do_exp
#         # correct
#         findfirst( x -> x>=real_time, cum_Δts)
#     else
        
#         return simulation_time = floor(Int,real_time / Δt + 1e-13)+1
#     end
# end
# -

function to_real_time(simulation_time::Int64, Δt::Float64; do_exp = false, cum_Δts = []) # from simulation time
    if do_exp
        return real_time = floor(Int,cum_Δts[simulation_time])
    else
        return real_time = floor(Int,(simulation_time-1) * Δt)
    end
end

function to_simulation_time(real_time, Δt::Float64;do_exp=false,cum_Δts=[]) #from real time 
    if do_exp
        findfirst( x -> x>real_time, cum_Δts)-1
    else
        
        frac = real_time * 1.0 / Δt
        return simulation_time = ceil(Int,frac+eps(frac))
    end
end

# # Testing simulation time stuff

if false

# +
Δt = 2.0
cum_times = vcat(0.0,cumsum(repeat([Δt],10)))

to_re(s) = to_real_time(s,Δt;do_exp=true,cum_Δts=cum_times) 
to_r(s) = to_real_time(s,Δt;do_exp=false,cum_Δts=cum_times) 
to_se(r) = to_simulation_time(r,Δt;do_exp=true,cum_Δts=cum_times) 
to_s(r) = to_simulation_time(r,Δt;do_exp=false,cum_Δts=cum_times) 
# -

s = [1,2,3]
r = to_r.(s)

to_s.(r) .- s

r = [0.2,2.2,4.2]
s = to_s.(r)

to_r.(s) .- floor.(r)

s = [1,2,3]
r = to_re.(s)

to_se.(r) .- s

r = [0.0,2.2,4.3]
s = to_s.(r)

to_re.(s) 

r = [0,2,4]
s = to_s.(r)

# +
# function to_real_time(simulation_time::Int64, Δt::Float64; do_exp = false, cum_Δts = []) # from simulation time
#     if do_exp
        
#         # correct
#         return real_time = floor(Int,cum_Δts[simulation_time])
#         ### return real = cum_Δts[simulation_time] ###
#     else
#         # correct
#         return real_time = floor(Int,(simulation_time-1) * Δt)
#     end
# end

# +
# function to_simulation_time(real_time, Δt::Float64;do_exp=false,cum_Δts=[]) #from real time 
#     if do_exp
#         # correct
#         findfirst( x -> x>real_time, cum_Δts)-1
#     else
        
#         frac = real_time * 1.0 / Δt
#         #return simulation_time = ceil(Int,frac )
#         return simulation_time = ceil(Int,frac+eps(frac))
#     end
# end

# +
Δt = 1.0#sqrt(11)/9
T = 10000
N = to_simulation_time(T,Δt;do_exp=false)
Δts = repeat([Δt],N)
Δts_cum = vcat(0:N) .* Δt

to_s_  = (r) -> to_simulation_time(r, Δt; do_exp = false, cum_Δts = Δts_cum)
to_s_e = (r) -> to_simulation_time(r, Δt; do_exp = true,  cum_Δts = Δts_cum)

my_real_times = vcat(0:T)
sample_inds_ = to_s_.(my_real_times)
sample_inds_e = to_s_e.(my_real_times)
sample_inds_ .- sample_inds_e

# +
Δt = sqrt(2)/2
T = 100
N = to_simulation_time(T,Δt;do_exp=false)
Δts = repeat([Δt],N)
Δts_cum = vcat(0:N) .* Δt

to_r_  = (s) -> to_real_time(s, Δt; do_exp = false, cum_Δts = Δts_cum)
to_r_e = (s) -> to_real_time(s, Δt; do_exp = true, cum_Δts = Δts_cum)

my_sim_times = 1:T
sample_inds_ = to_r_.(my_sim_times)
sample_inds_e = to_r_e.(my_sim_times)
sample_inds_ .- sample_inds_e
# -

end
