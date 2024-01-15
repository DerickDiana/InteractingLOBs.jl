# -*- coding: utf-8 -*-
using Revise
using Plots
using RelevanceStacktrace
using StatsBase 
using Distributions
using ProgressMeter
using Statistics
using Random  
using CurveFit
import Random:rand
using SpecialFunctions
using JLD2
using Combinatorics
using JET
using JLD2

using InteractingLOBs

Revise.revise()
testing = true
test_folder = "CheckData as of June 29"
seed = 942

function compare_objects(current_object,name_of_object_to_load; threshold = 1e-11, exact = false)
    name = string(test_folder,"/",name_of_object_to_load)
    compare = load_object(name)
    
    if exact
        return all(x->x, current_object .== compare)
    else
        diff = maximum(current_object .- compare)
        return abs(diff) < threshold
    end
end

# # TEST IF WORKING WITH TWO ORDER BOOKS WHICH ARE COUPLED BUT NO ANOMALOUS DIFFUSION

# +
# Configuration Arguments
num_paths = 1#30

L = 200#Int(floor(200*1))    # real system width (e.g. 200 meters)
M = 400#Int(floor(400*1))     # divided into M pieces , 400

p₀ = 1230.0  #this is the mid_price at t=0  238.75 
T = 5000

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 3.0; μ = 0.1; mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 6.0;  b = 1.5;  c = 1.2; myCouplingTerm = CouplingTerm(μ, a, b, c, true);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.1#0.95 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)


target_Δx = L / M  # real gap between simulation points 
Δt = (r * (target_Δx^2) / (2.0 * D))^(1/γ)

# RL Stuff:
RealKickStartTime = 50 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
SimKickEndTime = SimKickStartTime + 3 # when to stop kicking, in simulation time
#Position = 220
Position = -2
Volume = -8; # If position == -x where x>=0, then put it x above the mid price each time

myRLPusher1 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)


myRLPusher2 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)

lob_model¹ = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher1,myRandomnessTerm;do_interp=true,store_past_densities=true,do_cyclic_boundary=false,
    shift=0,Δx=target_Δx,do_exp_dist_times=true,do_test=true); #all this is to make an old test work with new code

lob_model² = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
    mySourceTerm, myCouplingTerm, myRLPusher2, myRandomnessTerm;do_interp=false,store_past_densities=true,
    shift=0,Δx=target_Δx,do_exp_dist_times=true,do_test=true); #all this is to make an old test work with new code

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model¹.SK_DP

# +
# clear everything pointed to by the dictionary then garbage collect. If it wasn't assigned yet it will left you know
try clear_double_dict(Dat) catch e print("Not initialized") end
GC.gc()

(Dat,break_points) = InteractOrderBooks([lob_model¹,lob_model²], seed, true; return_break_points=true) ;
#@report_opt InteractOrderBooks([lob_model¹], -1, true; return_break_points=true) 
# -

test1 = compare_objects(Dat[1][1].lob_densities,"CoupledNoAnomalousPsis",exact=true)
test2 = compare_objects(Dat[1][1].raw_price_paths,"CoupledNoAnomalousPrices",exact=true)
test1 && test2

# # TEST IF WORKING WITH TWO ORDER BOOKS WHICH ARE COUPLED BUT WITH ANOMALOUS DIFFUSION USING OLD CODE

# +
# Configuration Arguments
num_paths = 1#30

L = 200#Int(floor(200*1))    # real system width (e.g. 200 meters)
M = 400#Int(floor(400*1))     # divided into M pieces , 400

p₀ = 1230.0  #this is the mid_price at t=0  238.75 
T = 5000

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 0.8 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 3.0; μ = 0.1; mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 6.0;  b = 1.5;  c = 1.2; myCouplingTerm = CouplingTerm(μ, a, b, c, true);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.1#0.95 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)


target_Δx = L / M  # real gap between simulation points 
Δt = (r * (target_Δx^2) / (2.0 * D))^(1/γ)

# RL Stuff:
RealKickStartTime = 50 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
SimKickEndTime = SimKickStartTime + 3 # when to stop kicking, in simulation time
#Position = 220
Position = -2
Volume = -8; # If position == -x where x>=0, then put it x above the mid price each time

myRLPusher1 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)


myRLPusher2 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)

lob_model¹ = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher1,myRandomnessTerm;do_interp=true,store_past_densities=true,
    shift=0,Δx=target_Δx,kernel_cut_off=0.0045,do_test=false,do_exp_dist_times=false); #all this is to make an old test work with new code

lob_model² = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
    mySourceTerm, myCouplingTerm, myRLPusher2, myRandomnessTerm;do_interp=false,store_past_densities=true,
    shift=0,Δx=target_Δx,kernel_cut_off=0.0045,do_test=false,do_exp_dist_times=false); #all this is to make an old test work with new code

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model¹.SK_DP

# +
# clear everything pointed to by the dictionary then garbage collect. If it wasn't assigned yet it will left you know
try clear_double_dict(Dat) catch e print("Not initialized") end
GC.gc()

(Dat,break_points) = InteractOrderBooks([lob_model¹,lob_model²], seed, true; return_break_points=true) ;
# -

test3 = compare_objects(Dat[1][1].lob_densities,"CoupledAnomalousPsis";exact=true)
test4 = compare_objects(Dat[1][1].raw_price_paths,"CoupledAnomalousPrices";exact=true)
test3 && test4

# # TEST IF WORKING WITH TWO ORDER BOOKS WHICH ARE COUPLED BUT WITH ANOMALOUS DIFFUSION USING EXP CODE

# +
# Configuration Arguments
num_paths = 1#30

L = 200#Int(floor(200*1))    # real system width (e.g. 200 meters)
M = 400#Int(floor(400*1))     # divided into M pieces , 400

p₀ = 1230.0  #this is the mid_price at t=0  238.75 
T = 5000

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 0.8 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 3.0; μ = 0.1; mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 6.0;  b = 1.5;  c = 1.2; myCouplingTerm = CouplingTerm(μ, a, b, c, true);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.1#0.95 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)


target_Δx = L / M  # real gap between simulation points 
Δt = (r * (target_Δx^2) / (2.0 * D))^(1/γ)

# RL Stuff:
RealKickStartTime = 50 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
SimKickEndTime = SimKickStartTime + 3 # when to stop kicking, in simulation time
#Position = 220
Position = -2
Volume = -8; # If position == -x where x>=0, then put it x above the mid price each time

myRLPusher1 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)


myRLPusher2 = RLPushTerm(SimKickStartTime,SimKickEndTime,Position,Volume,false,false)

lob_model¹ = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher1,myRandomnessTerm;do_interp=true,store_past_densities=true,
    shift=0,Δx=target_Δx,kernel_cut_off=0.0045,do_test=true,do_exp_dist_times=true); #all this is to make an old test work with new code

lob_model² = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
    mySourceTerm, myCouplingTerm, myRLPusher2, myRandomnessTerm;do_interp=false,store_past_densities=true,
    shift=0,Δx=target_Δx,kernel_cut_off=0.0045,do_test=true,do_exp_dist_times=true); #all this is to make an old test work with new code

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model¹.SK_DP

# +
# clear everything pointed to by the dictionary then garbage collect. If it wasn't assigned yet it will left you know
try clear_double_dict(Dat) catch e print("Not initialized") end
GC.gc()

(Dat,break_points) = InteractOrderBooks([lob_model¹,lob_model²], seed, true; return_break_points=true) ;
# -

test5 = compare_objects(Dat[1][1].lob_densities,"CoupledAnomalousPsis";threshold=1e-13)
test6 = compare_objects(Dat[1][1].raw_price_paths,"CoupledAnomalousPrices";threshold=1e-300)
test5 && test6

# # Final results

# +
function print_result(outcome, number)
    print("Test ",number," is ",outcome ? "passed" : "failed","\n")
end

outcomes = [test1,test2,test3,test4,test5,test6]

for i in 1:length(outcomes)
    print_result(outcomes[i],i)
end

if all(outcomes)
    print("\nYaaaaay!\n")
else
    print("\nNooooooooooooo\n")
end
