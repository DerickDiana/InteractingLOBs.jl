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
import StatsBase:autocor
using SpecialFunctions
using LaTeXStrings
using JLD2
using Combinatorics
using JET
using Extremes, Dates, DataFrames, Distributions, StatsAPI 
using Formatting
using Printf
using Measurements
using LsqFit

include("../../ContinuousLearning.jl/src/ContinuousLearning.jl")
# one day can use: using ContinuousLearning 
# after: ] add "/home/derickdiana/Desktop/Masters/OfficialOne/ContinuousLearning.jl"
# but for now it's annoying to need to commit changes

using InteractingLOBs

# +
Revise.revise()


global_folder_path = "/home/derickdiana/Desktop/Masters/"
dat_folder_name = "ReportData"
picture_folder_name = "ReportImages"

#side_dimension = 4000
#my_size = 600 #essentially sets the scale
#my_dpi = (side_dimension)/(my_size) * 100

my_size = (600,600)
my_page_side = 3000
my_dpi = 200

# +
# Configuration Arguments
num_paths = 1#50#30

L = 300     # real system width (e.g. 200 meters)
M = 2*L     # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS) 
#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.7 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(μ, 0.0, 0.0, 0.0, false);

#seed = 3535956730
#seeds = repeat([seed],3)
seed = -1

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.9 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
kernel_cut = 0.00008

# RL Stuff:
T = 50
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10.0#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; 
        Δx=target_Δx, kernel_cut_off=kernel_cut)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; 
        Δx=target_Δx, kernel_cut_off=kernel_cut)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP

# +
len = 8

inv_range = range(0,1,length=len)
time_range = range(0,1,length=len)

#####################################################################################
#c = collect(Iterators.product(x_range,∂x_range,θ_range,∂θ_range));
c = collect(Iterators.product(inv_range,time_range));
c = ((x)->collect(x)).(c)

tempf = (f) -> (f)
inv_range_s  =  repeat([tempf(  (inv_range[end]-inv_range[1])/len)],len)
time_range_s =  repeat([tempf(  (time_range[end]-time_range[1])/len)],len )

#####################################################################################
s = collect(Iterators.product(inv_range_s,time_range_s));
s = ((x)->collect(x)).(s)

# +
# Actor
initial_w_scale = 0.0
w_A = (rand(length(c)).*initial_w_scale*2).-initial_w_scale
A = 0.0
η_A = 10.0
n_A = 0.0

# Critic
initial_w_scale = 0.0
w_V = (rand(length(c)).*initial_w_scale*2).-initial_w_scale
V = 0.0
Vmax = +1.0
Vmin = -0.0
e = repeat([0.0],length(w_V))
η_V = 5.0 

τ = 2.0
κ = τ/10

if τ==Δt
    @assert κ == τ 
    
    γ = 0.0
else
    γ=(1-Δt/κ)/(1-Δt/τ)
end
λ=(1-Δt/τ);

σ_0 = 2.0

init_brain = RLBrain(w_A,w_V,e,A,V,σ,n_A);

final_brain = InteractingLOBs.ContinuousLearning.copy_brain(init_brain)
# -

for i in 1:1#200
    RLParam_ = RLParam(c,s,η_A,η_V,Vmax,Vmin,γ,λ,τ,σ_0,Δt,final_brain);

    (Dat,p_arr) = quick_plot([lob_model_push],[RLParam_],SimKickStartTime-1,12,
        for_visual=(do_left=false,do_right=false))#,seed=seed)
    plot!()

    final_brain = InteractingLOBs.ContinuousLearning.copy_brain(
                                                    Dat[1][1].RLBrains[end]
                                                )
end

# +
plt = plot_price_path(plot(),Dat, 300, 1, false; path_to_plot = 1,do_scatter=true)

frac = 0.2
xb = 0.05
plt = plot!(plt, inset = (1,bbox(xb, xb, frac, frac))  )

plt = plot_price_path(plt, Dat, 16, 1, false; path_to_plot = 1,do_scatter=true, subplot=2, do_extra_bit=true)

ymin = ylims(plt[2])[1]
xmin = xlims(plt[2])[1]
ymax = ylims(plt[2])[2]
xmax = xlims(plt[2])[2]

my_y_ticks = ((v)->round(v,digits=2)).([ymin+0.01,(ymin+ymax)/2,ymax-0.01])
my_x_ticks = ((v)->round(v,digits=2)).([xmin+0.01,(xmin+xmax)/2,xmax-0.02])

plt = plot!(plt,subplot=2,mirror=true,xlab="",ylab="",xticks=my_x_ticks,ytick=my_y_ticks,frame=:box,
                xlims=[xmin,xmax-0.01],ylims=[ymin,ymax])

plt = draw_square_abs(plt,[xmin,ymin],[xmax,ymax])
plt = arrow_from_abs_to_frac(plt,[xmin,ymax],[xb,1-(xb+frac)])
plt = arrow_from_abs_to_frac(plt,[xmax,ymax],[xb+frac,1-(xb+frac)])
# -

show_path(Dat;kw_for_visual=(shift_back_approx=true,do_left=false,do_right=false),fps_target=5)

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "ManyPathVisual"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
