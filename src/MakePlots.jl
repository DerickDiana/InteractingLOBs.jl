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
#using Measures

using InteractingLOBs

using ContinuousLearning

# +
pm = Plots.PlotMeasures
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
# -

"""
    get_tickslogscale(lims; skiplog=false)
Return a tuple (ticks, ticklabels) for the axis limit `lims`
where multiples of 10 are major ticks with label and minor ticks have no label
skiplog argument should be set to true if `lims` is already in log scale.
"""
function get_tickslogscale(lims::Tuple{T, T}; skiplog::Bool=false) where {T<:AbstractFloat}
    mags = if skiplog
        # if the limits are already in log scale
        floor.(lims)
    else
        floor.(log10.(lims))
    end
    rlims = if skiplog; 10 .^(lims) else lims end

    total_tickvalues = []
    total_ticknames = []

    rgs = range(mags..., step=1)
    for (i, m) in enumerate(rgs)
        if m >= 0
            tickvalues = range(Int(10^m), Int(10^(m+1)); step=Int(10^m))
            ticknames  = vcat([string(round(Int, 10^(m)))],
                              ["" for i in 2:9],
                              [string(round(Int, 10^(m+1)))])
        else
            tickvalues = range(10^m, 10^(m+1); step=10^m)
            ticknames  = vcat([string(10^(m))], ["" for i in 2:9], [string(10^(m+1))])
        end

        if i==1
            # lower bound
            indexlb = findlast(x->x<rlims[1], tickvalues)
            if isnothing(indexlb); indexlb=1 end
        else
            indexlb = 1
        end
        if i==length(rgs)
            # higher bound
            indexhb = findfirst(x->x>rlims[2], tickvalues)
            if isnothing(indexhb); indexhb=10 end
        else
            # do not take the last index if not the last magnitude
            indexhb = 9
        end

        total_tickvalues = vcat(total_tickvalues, tickvalues[indexlb:indexhb])
        total_ticknames = vcat(total_ticknames, ticknames[indexlb:indexhb])
    end
    return (total_tickvalues, total_ticknames)
end

"""
    fancylogscale!(p; forcex=false, forcey=false)
Transform the ticks to log scale for the axis with scale=:log10.
forcex and forcey can be set to true to force the transformation
if the variable is already expressed in log10 units.
"""
function fancylogscale!(p::Plots.Subplot; forcex::Bool=false, forcey::Bool=false)
    kwargs = Dict()
    for (ax, force, lims) in zip((:x, :y), (forcex, forcey), (xlims, ylims))
        axis = Symbol("$(ax)axis")
        ticks = Symbol("$(ax)ticks")

        if force || p.attr[axis][:scale] == :log10
            # Get limits of the plot and convert to Float
            ls = float.(lims(p))
            ts = if force
                (vals, labs) = get_tickslogscale(ls; skiplog=true)
                (log10.(vals), labs)
            else
                get_tickslogscale(ls)
            end
            kwargs[ticks] = ts
        end
    end

    if length(kwargs) > 0
        plot!(p; kwargs...)
    end
    p
end
fancylogscale!(p::Plots.Plot; kwargs...) = (fancylogscale!(p.subplots[1]; kwargs...); return p)
fancylogscale!(; kwargs...) = fancylogscale!(plot!(); kwargs...)

function get_size_and_dpi(;target_dpi=-1,target_side=-1,fraction_of_page=1/2, target_page_side = 3000)
    # size refers to pixels
    # governing equation:
    # dpi * side / 100  = target_page_side * fraction_of_page
    # the target_side is for an entire page width. The height doesn't matter since we always target squares
    
    if target_dpi != -1 && target_side != -1
        @warn "Cannot set both dpi and size"
    end
    
    if target_dpi == -1
        target_dpi = target_page_side * 100 * fraction_of_page / target_side
    end
    
    if target_side == -1
        target_side = floor(Int64,target_page_side * 100 * fraction_of_page / target_dpi )
    end
    
    return (target_dpi, (target_side,target_side))
end

function my_compare(dats)
    print("σ   ","β   ","y   ","\n")
    for dat in dats
        σ_ = round(dat[1][1].slob.randomness_term.σ,digits=2)
        β_ = round(dat[1][1].slob.randomness_term.β,digits=2)
        γ_ = round(dat[1][1].slob.γ,digits=2)
        e_ = dat[1][1].slob.do_exp_dist_times ? "Exp" : "Unif"
        print(σ_," ",β_," ",γ_," ",e_,"\n")
    end
end

# # GENERAL WORKING

# ## Visualize LatOB resting state

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);


# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 4000
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP

# +
#ηδσΔκτₚₙₚₙη
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1,1,
    for_visual=(do_left=false,do_right=false,x_axis_width=20))
plot!()

subs = Dat[1][1].lob_densities[:,300:end-1]
#subs = log.(subs.+1)
plot(get_sums(subs))

# +
file_name = "V_LatOB_exp"
folder_path = string(global_folder_path,"/",picture_folder_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                folder_path = folder_path, file_name = file_name,folder_name = "Singles", 
                for_plot=(legendfontsize=12,),scale_=1.0)
plot!()
# -

# ## Visualize LOB resting state with exponential things

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1,1;
    for_visual=(x_axis_width=2,))
plot!()

# +
file_name = "V_LOB_exp"
folder_path = string(global_folder_path,"/",picture_folder_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                folder_path = folder_path, file_name = file_name,folder_name = "Singles", 
                for_plot=(legendfontsize=12,),scale_=1.0)
plot!()
# -

# ## Visualize LOB resting state

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1,1,
    for_visual=(do_left=false,do_right=false))
plot!()

# +
file_name = "V_LOB"
folder_path = string(global_folder_path,"/",picture_folder_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                folder_path = folder_path, file_name = file_name,folder_name = "Singles", 
                for_plot=(legendfontsize=12,),scale_=1.0)
plot!()
# -

# ## Normal kick

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false, beginning_shift_frac=0.5)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false, beginning_shift_frac=0.5)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1;
    for_visual=(do_left=false,do_right=false,x_axis_width=3))
plot!()

# +
file_name = "K"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Normal path

# +
# Configuration Arguments
num_paths = 1#50#30

L = 50     # real system width (e.g. 200 meters)
M = 5*L     # divided into M pieces , 400

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

seed = 3535956730
seeds = repeat([seed],3)

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
T = 100
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 1.0#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; 
        Δx=target_Δx, kernel_cut_off=kernel_cut)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; 
        Δx=target_Δx, kernel_cut_off=kernel_cut)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat3,p_arr) = quick_plot([lob_model_push,lob_model_no_push],[],SimKickStartTime-1,12,
    for_visual=(do_left=false,do_right=false))#,seed=seed)
plot!()
Δt3 = Δt

Dat = Dat3
plt = plot_price_path(plot(),Dat, 250, 1, false; path_to_plot = 1,do_scatter=true)

# +
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

show_path(Dat;kw_for_visual=(shift_back_approx=true,))

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "ManyPathVisual"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

d1 = Dat1[1][1]
l1,r1 = get_effective_market_orders(Dat1;shift=0)
l1 = .-l1
derivs1 = get_central_derivative(Dat1)
V = Dat[1][1].V;
r_ = d.slob.randomness_term.r

d2 = Dat2[1][1]
l2,r2 = get_effective_market_orders(Dat2;shift=0)
l2 = .-l2
derivs2 = get_central_derivative(Dat2)
V = Dat[1][1].V;
r_ = d.slob.randomness_term.r

d3 = Dat3[1][1]
l3,r3 = get_effective_market_orders(Dat3;shift=0)
l3 = .-l3
derivs3 = get_central_derivative(Dat3)
V = Dat[1][1].V;
r_ = d.slob.randomness_term.r

histogram(r1.+l1,alpha=0.5)

histogram(r2.+l2,alpha=0.5)

a = mean(r.+l)
b = maximum(r.+l)

change_in_sum1 = diff(get_sums(Dat1[1][1].lob_densities[:,2:end-1];absolute=true))
plot(change_in_sum1)

change_in_sum2 = diff(get_sums(Dat2[1][1].lob_densities[:,2:end-1];absolute=true))
plot(change_in_sum2)

plot()
plot!(  (D.*abs.(derivs1)*Δt1)[2:end-1]  ,label=L"D\partial_x \varphi(x,t) Δt |_{x=p_t}"         ,col=3)
#plot!(  (r .+ l     )[2:end-1]  ,label=L"r(t)+l(t)"   ,color=2)
scatter!(  ((r1 .+ l1)     )[2:end-1]  ,label=L"r(t)+l(t)"   ,color=2, markerstrokewidth=0.0, markersize=0.3)
plot!(legendfontsize=10,legend=:bottomright)
#plot!(change_in_sum.*Δt,ylims=[-0.01,0.04])

plot()
plot!(  (D.*abs.(derivs2)*Δt2)[2:end-1]  ,label=L"D\partial_x \varphi(x,t) Δt |_{x=p_t}"         ,col=3)
#plot!(  (r .+ l     )[2:end-1]  ,label=L"r(t)+l(t)"   ,color=2)
scatter!(  ((r2 .+ l2)     )[2:end-1]  ,label=L"r(t)+l(t)"   ,color=2, markerstrokewidth=0.0, markersize=0.3)
plot!(legendfontsize=10,legend=:bottomright)
#plot!(change_in_sum.*Δt,ylims=[-0.01,0.04])

plot()
len = length(cumsum(  D.*abs.(derivs1)*Δt1 )[2:end-1])
xrange1 = (1:len)./len*100
plot!( xrange1 , cumsum(  D.*abs.(derivs1)*Δt1 )[2:end-1]  ,label=L"\int_0^t 0.125 \times \partial_x \varphi(x,t) |_{x=p_t}"         ,col=3)
plot!( xrange1 , cumsum((r1 .+ l1)    )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
#scatter!(  cumsum(r .+ l     )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2,ms=3.0,markerstrokewidth=0.1,  ma=1)
#plot!(legendfontsize=10,legend=:bottomright)

plot()
len = length(cumsum(  D.*abs.(derivs2)*Δt2 )[2:end-1])
xrange2 = (1:len)./len*100
plot!( xrange2 , cumsum(  D.*abs.(derivs2)*Δt2 )[2:end-1]  ,label=L"\int_0^t 0.125 \times \partial_x \varphi(x,t) |_{x=p_t}"         ,col=3)
plot!( xrange2 , cumsum((r2 .+ l2)    )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
#plot!( xrange2 , (cumsum(r3 .+ l3) ./ cumsum(D.*abs.(derivs3)*Δt3))[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
#scatter!(  cumsum(r .+ l     )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2,ms=3.0,markerstrokewidth=0.1,  ma=1)
#plot!(legendfontsize=10,legend=:bottomright)

plot()
len = length(cumsum(  D.*abs.(derivs3)*Δt3 )[2:end-1])
xrange3 = (1:len)./len*100
#plot!( xrange , cumsum(  D.*abs.(derivs3)*Δt3 )[2:end-1]  ,label=L"\int_0^t 0.125 \times \partial_x \varphi(x,t) |_{x=p_t}"         ,col=3)
#plot!( xrange , cumsum((r3 .+ l3)    )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
plot!( xrange3 , (cumsum(r3 .+ l3) ./ cumsum(D.*abs.(derivs3)*Δt3))[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
#scatter!(  cumsum(r .+ l     )[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2,ms=3.0,markerstrokewidth=0.1,  ma=1)
#plot!(legendfontsize=10,legend=:bottomright)

plot()
plot!( xrange1 , (cumsum(r1 .+ l1) ./ cumsum(D.*abs.(derivs1)*Δt1))[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
plot!( xrange2 , (cumsum(r2 .+ l2) ./ cumsum(D.*abs.(derivs2)*Δt2))[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
plot!( xrange3 , (cumsum(r3 .+ l3) ./ cumsum(D.*abs.(derivs3)*Δt3))[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"   ,color=2)
plot!(legend=:none)

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "OptionsForVolumeTradedSum"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

histogram(r.-l,alpha=0.5)
histogram!(V./270,alpha=0.5)

a = mean(r.-l)
b = maximum(r.-l)

plot()
plot!(  (V./250          )[2:end-1]  ,label=L"0.005 V(t)"               ,col=1)
plot!(  (r .- l     )[2:end-1]  ,label=L"r(t)-l(t)"   ,col=2)
plot!(legendfontsize=10,legend=:bottomright)

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "OptionsForVolumeTradedDiff"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)

# +
#plot()
#plot!(  (abs.(V./66)          )[2:end-1]  ,label="Value of V(t)"               ,col=1)
#plot!(  (abs.(l).+abs.(r)      )[2:end-1]  ,label="Sum of MO by cancellation"   ,col=2)
#plot!(  (abs.(derivs./15)      )[2:end-1]  ,label="Derivative at price"         ,col=3)
#plot!(  (abs.(abs.(l).-abs.(r))  )[2:end-1]  ,label="Diff of MO by cancellation"  ,col=4)
#plot!(  (Dat[1][1].P⁺s./10             )[2:end-1]  ,label="Diff of MO by cancellation"  ,col=4)
#plot!(  (l      )[2:end-1]  ,label="Sum of MO by cancellation"   ,col=2)
#plot!(  (r      )[2:end-1]  ,label="Sum of MO by cancellation"   ,col=2)

#cor(abs.(V./200)[2:end],(abs.(derivs./15)))
# -

plot()
plot!(  (V./200          )[2:end-1]  ,label=L"0.005 V(t)"               ,col=1)
#plot!(  (abs.(derivs./8)      )[2:end-1]  ,label=L"0.125 \times \partial_x \varphi(x,t) |_{x=p_t}"         ,col=3)
#plot!(  (l      )[2:end-1]  ,label="Sells jump left"   ,col=2)
#plot!(  (r      )[2:end-1]  ,label="Buys jump right"   ,col=2)
plot!(  (r .- l     )[2:end-1]  ,label=L"r(t)-l(t)"   ,col=2)
#plot!(  (r .+ l     )[2:end-1]  ,label=L"r(t)+l(t)"   ,col=2)
#plot!(  ((d.P⁺s .- d.P⁻s)./10)[2:end-1]  ,label="Difference in probability"   ,col=2)
#plot!(  (-diff(d.raw_price_paths)./10)[2:end-1]  ,label="Change in price"   ,col=2)
plot!(legendfontsize=10,legend=:bottomright)

for (a,b) in ([1,2],[4,5])
    print(a," ",b)
end

# ## Trying stuff

# +
# Configuration Arguments
plot()
L = 60
result = zeros(Float64,20)

r = 0.1
#for (M,ν,μ,σ) in ([10*L,0.5,0.7,0.7],[10*L,0.8,0.5,0.4],[10*L,1.2,0.9,0.7])
for i in 1:20
    num_paths = 1#50#30
    
    (ν,μ,σ) = (rand(), rand(), rand())
    print((ν," ",μ," ",σ,"\n"))

    L = 60     # real system width (e.g. 200 meters)
    M = 10*L     # divided into M pieces , 400

    p₀ = 1300.0  #this is the mid_price at t=0  238.75 

    # Free-Parameters for gaussian version
    D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
    α = 0.0 # legacy, no longer used

    #ν = 0.5 #removal rate
    γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS) 
    #ν = 1.0#1.0#3.0 #removal rate
    #γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

    # Source term:
    λ = 1.0 #
    #μ = 0.7 #
    mySourceTerm = SourceTerm(λ, μ, true);

    # Coupling term:
    myCouplingTerm = CouplingTerm(μ, 0.0, 0.0, 0.0, false);

    #seed = 3535956730
    #seeds = repeat([seed],3)

    # My randomness term
    σ = 1.0 #variance in randomness
    β = 0.9 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = true #behave like a random walk
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)

    target_Δx = L / M  # real gap between simulation points 
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
    kernel_cut = 0.00008

    # RL Stuff:
    T = 100
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
    
    Dat = InteractOrderBooks([lob_model_push,lob_model_no_push], -1, true) ;
    
    d1 = Dat[1][1]
    l_,r_ = get_effective_market_orders(Dat;shift=0)
    l_ = .-l_
    derivs = get_central_derivative(Dat)
    
    x = r_ .+ l_
    y = D.*abs.(derivs)*Δt
    
    cumx = cumsum(x)
    cumy = cumsum(y)
    
    
    
    len = length(cumsum(  D.*abs.(derivs)*Δt )[2:end-1])
    xrange = (1:len)./len*100
    rat = cumx ./ cumy
    print(rat[end-1],"\n")
    result[i] = rat[end-1]
    
    #plot!( xrange , (rat)[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"  )
    
end
plot!(legend=:none)

# +
# Configuration Arguments
plot()
L = 60
result = zeros(Float64,1)

r = 0.4
#for (M,ν,μ,σ) in ([10*L,0.5,0.7,0.7],[10*L,0.8,0.5,0.4],[10*L,1.2,0.9,0.7])
for i in 1
    num_paths = 1#50#30
    
    (ν,μ,σ) = (rand(), rand(), rand())
    print((ν," ",μ," ",σ,"\n"))

    L = 60     # real system width (e.g. 200 meters)
    M = 20*L     # divided into M pieces , 400

    p₀ = 1300.0  #this is the mid_price at t=0  238.75 

    # Free-Parameters for gaussian version
    D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
    α = 0.0 # legacy, no longer used

    #ν = 0.5 #removal rate
    γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS) 
    #ν = 1.0#1.0#3.0 #removal rate
    #γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

    # Source term:
    λ = 1.0 #
    #μ = 0.7 #
    mySourceTerm = SourceTerm(λ, μ, true);

    # Coupling term:
    myCouplingTerm = CouplingTerm(μ, 0.0, 0.0, 0.0, false);

    #seed = 3535956730
    #seeds = repeat([seed],3)

    # My randomness term
    σ = 1.0 #variance in randomness
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
    
    Dat = InteractOrderBooks([lob_model_push,lob_model_no_push], -1, true) ;
    
    d1 = Dat[1][1]
    l_,r_ = get_effective_market_orders(Dat;shift=0)
    l_ = .-l_
    derivs = get_central_derivative(Dat)
    
    x = 1/(2 * (1-r) / (2-r)) .*(r_ .+ l_)
    y =  D.*abs.(derivs)*Δt
    
    cumx = cumsum(x)
    cumy = cumsum(y)
    
    len = length(cumx[2:end-1])
    xrange = (1:len)./len*100
    rat = cumx ./ cumy
    
    scatter!(xrange,x[2:end-1])
    plot!(xrange,y[2:end-1])
    print(rat[end-1],"\n")
    result[i] = rat[end-1]
    
    #plot!( xrange , (rat)[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"  )
    
end
plot!(legend=:none)
# -

histogram(x,alpha=0.5,bin=300)
vline!( [y[end-1]], lw=5)
vline!( [mean(x)], color="green", xticks = range(15e-6,20e-6,length=3))

sort(x)[end]

h = StatsBase.fit(Histogram, x, nbins=300).edges[1][end-1]

length(x[x.>1.78e-5])/length(x[x.<1.78e-5])

print(mean(result))
histogram(result)
vline!( [2 * (1-r) / (2-r)] , lw=3)

plot(rat)

plot()
plot!(xrange,cumx[2:end-1])
plot!(xrange,(2 * (1-r) / (2-r)).*cumy[2:end-1])

# +
# Configuration Arguments
plot()
L = 60
for (M,ν,μ,σ) in ([15*L,0.5,0.7,0.7],[20*L,0.8,0.5,0.4],[25*L,1.2,0.9,0.7])
    num_paths = 1#50#30

    L = 60     # real system width (e.g. 200 meters)
    #M = 5*L     # divided into M pieces , 400

    p₀ = 1300.0  #this is the mid_price at t=0  238.75 

    # Free-Parameters for gaussian version
    D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
    α = 0.0 # legacy, no longer used

    #ν = 0.5 #removal rate
    γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS) 
    #ν = 1.0#1.0#3.0 #removal rate
    #γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

    # Source term:
    λ = 1.0 #
    #μ = 0.7 #
    mySourceTerm = SourceTerm(λ, μ, true);

    # Coupling term:
    myCouplingTerm = CouplingTerm(μ, 0.0, 0.0, 0.0, false);

    seed = 3535956730
    seeds = repeat([seed],3)

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
    T = 100
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
    
    Dat = InteractOrderBooks([lob_model_push,lob_model_no_push], seed, true) ;
    
    l,r = get_effective_market_orders(Dat;shift=0)
    l = .-l
    derivs = get_central_derivative(Dat)
    
    x = r .+ l
    y = D.*abs.(derivs)*Δt
    
    cumx = cumsum(x)
    cumy = cumsum(y)
    
    len = length(cumsum(  D.*abs.(derivs)*Δt )[2:end-1])
    xrange = (1:len)./len*100
    
    
    histogram!(r.+l,alpha=0.5)
    
#     rat = 3/2 .* cumx ./ cumy
#     print(rat[end-1],"\n")
    
#     plot!( xrange , (rat)[2:end-1]  ,label=L"\int_0^t r(t)+l(t)"  )
    
end
plot!(legend=:none)
# -

# ## Kick with interpolation

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false, do_interp = true)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false, do_interp = true)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1;
    for_visual=(do_left=false,do_right=false))
plot!()

# +
file_name = "K_interp"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Kick using exp dist times

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#0.3#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1;
    for_visual=(x_axis_width=3,shift_back_approx=true,kw_for_plot=(legend=:none,)))
plot!()

# +
file_name = "K_exp"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Kick using exp dist times with interpolation

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 10#0.3#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; 
        Δx=target_Δx, do_exp_dist_times=true,do_interp=true)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; 
        Δx=target_Δx, do_exp_dist_times=true,do_interp=true)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1;
    for_visual=(x_axis_width=3,shift_back_approx=true,kw_for_plot=()))
plot!()

# +
file_name = "K_exp_interp"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Kick using exp dist times with market order

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = 0.3#10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,true)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=true)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime-1;
    for_visual=(x_axis_width=3,shift_back_approx=true,kw_for_plot=()))
plot!()

# +
file_name = "K_exp_MO"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Kick using market order

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = exp(-3)
#Volume = exp(0)

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,true)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; 
    Δx=target_Δx, do_exp_dist_times=false,do_interp=false)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; 
    Δx=target_Δx, do_exp_dist_times=false,do_interp=false)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime+1,2;
    for_visual=(
        x_axis_width=5,shift_back_approx=true,
        do_left=false,do_right=false,annotate_pos=:topright,
        kw_for_plot=(legend=:bottomleft,)
        ))
plot!()

# +
file_name = "K_MO"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# ## Meta order visual

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 1.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -3 
Volume = exp(-0.06)
Volume = exp(0)
Volume = exp(-3)

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+10,Position,Volume,false,true)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+10,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP

# +
(Dat,p_arr) = quick_plot([lob_model_push],[],SimKickStartTime+1;
    for_visual=(
        x_axis_width=10,shift_back_approx=true,
        do_left=false,do_right=false,annotate_pos=:topright,
        kw_for_plot=(legend=:bottomleft,)
        ))
plot!()

#savefig(plot!(),"/home/derickdiana/Desktop/MetaOrderVisual.png")
# -

plot(abs.(Dat[1][1].raw_price_paths[SimKickStartTime-5:SimKickStartTime+100]),xlabel="Time",ylabel="Price",label="Price")
vline!([5+1],label="Start of meta order")
vline!([10+5+1],label="End of meta order")
#savefig(plot!(),"/home/derickdiana/Desktop/PriceOverTime.png")

# +
file_name = "K_MO"

folder_path = string(global_folder_path,picture_folder_name,"/","Visualisations")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/4)
names = repeat([file_name],length(p_arr))

save_figs(p_arr;folder_name=file_name,names=names,folder_path=folder_path,save_type="png",
            dpi=dpi_,plot_size=size_,scale_=1.0,notify=true,insert_numbers=true)
# -

# # STYLIZED FACTS

# ## Different Diffusions with same Random Walk and Variance

# +
# Configuration Arguments
T = 200#3600*8
num_paths = 1 

L = 1000     # real system width (e.g. 200 meters)
M = 2*L    # divided into M pieces , 400

p₀ = 1300.0 ###1230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate

# Source term:
λ = 1.0
μ = 0.1 


mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);
seed = 3535956730

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,0.9,lag,do_random_walk,true)

target_Δx = L / M  # real gap between simulation points 


γs = [1.0, 0.8, 0.6] #######

myRLPusher = RLPushTerm(0.0,0.0,0.0,0.0,false,false)
kernel_cut = 0.00008

Δt_ = calculate_Δt_from_Δx(target_Δx,γs[1],D,r)
lob_model_1 = SLOB(num_paths, T, p₀, Δt_, L, D, ν, α, γs[1], 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);

Δt_ = calculate_Δt_from_Δx(target_Δx,γs[2],D,r)
lob_model_2 = SLOB(num_paths, T, p₀, Δt_, L, D, ν, α, γs[2], 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);

Δt_ = calculate_Δt_from_Δx(target_Δx,γs[3],D,r)
lob_model_3 = SLOB(num_paths, T, p₀, Δt_, L, D, ν, α, γs[3], 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);


seeds = repeat([seed],3)

print((Δt_,to_simulation_time(T,Δt_),num_paths*to_simulation_time(T,Δt_))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_2.SK_DP

# +
# my_file_name = "SF_DD"

# my_print_labels = [string("\nDo run with γ=",γs[1],"\n"),
#                    string("\nDo run with γ=",γs[2],"\n"),
#                    string("\nDo run with γ=",γs[3],"\n")]

# seeds = repeat([seed],3)
    
# (Dat1,Dat2,Dat3) = obtain_data_list([lob_model_3,lob_model_2,lob_model_1]
#                                         ;seeds=seeds,do_new=false,print_labels=my_print_labels,save_new=true,
#                                          folder_name=dat_folder_name,file_name=my_file_name,
#                                          folder_path=global_folder_path);
# (Dat3,Dat2,Dat1) = (Dat1,Dat2,Dat3);
# -

my_file_name = "SF_DD_3"
(Dat3,) = obtain_data_list([lob_model_3]
                            ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                             folder_name=dat_folder_name,file_name=my_file_name,
                             folder_path=global_folder_path);

my_file_name = "SF_DD_1"
(Dat1,) = obtain_data_list([lob_model_1]
                            ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                             folder_name=dat_folder_name,file_name=my_file_name,
                             folder_path=global_folder_path);

my_file_name = "SF_DD_2"
(Dat2,) = obtain_data_list([lob_model_2]
                            ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                             folder_name=dat_folder_name,file_name=my_file_name,
                             folder_path=global_folder_path);

length(Dat2[1][1].raw_price_paths)

l = floor(Int64,1/Dat2[1][1].slob.Δt)

length(Dat1[1][1].raw_price_paths)

Dat1[1][1].raw_price_paths = Dat1[1][1].raw_price_paths[1:l:end]
Dat2[1][1].raw_price_paths = Dat2[1][1].raw_price_paths[1:l:end]
Dat3[1][1].raw_price_paths = Dat3[1][1].raw_price_paths[1:l:end];

length(Dat1[1][1].raw_price_paths)

# +
sf = StylizedFactsPlot([Dat1,Dat2,Dat3];do_raw=false)#,real_upper_cut_off=30)#25000);

plot_all_stylized_facts(sf,plot_size=(2000,1000);tolerance=3.0)

# +
my_file_name = "SF_DD"
folder_name = string(picture_folder_name,"/",my_file_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/3)

save_seperate(sf;folder_name=folder_name,file_name=my_file_name,folder_path=global_folder_path,save_type="png",
    scale_=1.0,plot_size=size_,dpi=dpi_,tolerance=1.0,
    for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=1.0))
# -

# ## Different Correlation with same Diffusion and Variance

# +
# Configuration Arguments
T = 200#3600*8#86400 
num_paths = 1 

L = 1000     # real system width (e.g. 200 meters)
M = 2*L    # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate

# Source term:
λ = 1.0
μ = 0.1 
γ = 0.8

mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);
seed = 3535956730

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
lag = 10 #lag
do_random_walk = false #behave like a random walk
corrs = [0.9,0.8,-0.1]
myRandomnessTerm_1 = RandomnessTerm(σ,r,corrs[1],lag,true,true)
myRandomnessTerm_2 = RandomnessTerm(σ,r,corrs[2],lag,true,true)
myRandomnessTerm_3 = RandomnessTerm(σ,r,corrs[3],lag,false,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)


myRLPusher = RLPushTerm(0.0,0.0,0.0,0.0,false,false)
kernel_cut = 0.00008

lob_model_1 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_1; store_past_densities=false, 
    Δx =  target_Δx, seed=seed, kernel_cut_off = kernel_cut);
lob_model_2 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_2; store_past_densities=false, 
    Δx =  target_Δx, seed=seed, kernel_cut_off = kernel_cut);
lob_model_3 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_3; store_past_densities=false, 
    Δx =  target_Δx, seed=seed, kernel_cut_off = kernel_cut);

seeds = repeat([seed],3)

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_1.SK_DP

# +
# my_file_name = "SF_DC"

# my_print_labels = [string("\nDo run with corr=",corrs[1],"\n"),
#                    string("\nDo run with corr=",corrs[2],"\n"),
#                    string("\nDo run with corr=",corrs[3],"\n")]

# seeds = repeat([seed],3)
    
# (Dat1,Dat2,Dat3) = obtain_data_list([lob_model_1,lob_model_2,lob_model_3]
#                                         ;seeds=seeds,do_new=false,print_labels=my_print_labels,save_new=true,
#                                          folder_name=dat_folder_name,file_name=my_file_name,
#                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DC_1"
    
(Dat1,) = obtain_data_list([lob_model_1]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DC_2"
    
(Dat2,) = obtain_data_list([lob_model_2]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DC_3"
    
(Dat3,) = obtain_data_list([lob_model_3]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);
# -

my_compare([Dat1,Dat2,Dat3])

# +
sf = StylizedFactsPlot([Dat2,Dat1,Dat3];do_raw=false)#,real_upper_cut_off=25000);

plot_all_stylized_facts(sf,plot_size=(2000,1000);tolerance=1.0,
for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=0.5))

# +
my_file_name = "SF_DC"
folder_name = string(picture_folder_name,"/",my_file_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/3)

save_seperate(sf;folder_name=folder_name,file_name=my_file_name,folder_path=global_folder_path,save_type="png",
    scale_=1.0,plot_size=size_,dpi=dpi_,tolerance=[1.7,1.0,1.0],quick=false,
    for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=0.5))
# -

# ## Different Variances with same Random Walk and Diffusion

# +
# Configuration Arguments
T = 200#3600*8#86400
num_paths = 1 

L = 1000     # real system width (e.g. 200 meters)
M = 2*L    # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate

# Source term:
λ = 1.0
μ = 0.1 
γ = 0.8

mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);
#seed = 3537480
seed = 3535956730

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
lag = 10 #lag
do_random_walk = false #behave like a random walk
σs = [0.5,1.0,1.5]
myRandomnessTerm_1 = RandomnessTerm(σs[1],r,0.9,lag,true,true)
myRandomnessTerm_2 = RandomnessTerm(σs[2],r,0.9,lag,true,true)
myRandomnessTerm_3 = RandomnessTerm(σs[3],r,0.9,lag,true,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)


myRLPusher = RLPushTerm(0.0,0.0,0.0,0.0,false,false)
kernel_cut = 0.00008

lob_model_1 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_1; store_past_densities=false, 
    Δx =  target_Δx, seed=seed,kernel_cut_off=kernel_cut);
lob_model_2 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_2; store_past_densities=false, 
    Δx =  target_Δx, seed=seed,kernel_cut_off=kernel_cut);
lob_model_3 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_3; store_past_densities=false, 
    Δx =  target_Δx, seed=seed,kernel_cut_off=kernel_cut);


seeds = repeat([seed],3)

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_1.SK_DP

# +
# my_file_name = "SF_DV"

# my_print_labels = [string("\nDo run with σ=",σs[1],"\n"),
#                    string("\nDo run with σ=",σs[2],"\n"),
#                    string("\nDo run with σ=",σs[3],"\n")]

# seeds = repeat([seed],3)
    
# (Dat1,Dat2,Dat3) = obtain_data_list([lob_model_1,lob_model_2,lob_model_3]
#                                         ;seeds=seeds,do_new=false,print_labels=my_print_labels,save_new=true,
#                                          folder_name=dat_folder_name,file_name=my_file_name,
#                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DV_1"
    
(Dat1,) = obtain_data_list([lob_model_1]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DD_2"
    
(Dat2,) = obtain_data_list([lob_model_2]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DV_3"
    
(Dat3,) = obtain_data_list([lob_model_3]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
sf = StylizedFactsPlot([Dat1,Dat2,Dat3];do_raw=false)#,real_upper_cut_off = 25000);

plot_all_stylized_facts(sf,plot_size=(2000,1000);tolerance=1.0)

# +
my_file_name = "SF_DV"
folder_name = string(picture_folder_name,"/",my_file_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/3)

save_seperate(sf;folder_name=folder_name,file_name=my_file_name,folder_path=global_folder_path,save_type="png",
    scale_=1.0,plot_size=size_,dpi=dpi_,tolerance=1.0,
    for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=0.5))
# -

# ## Different Times

# +
# Configuration Arguments
T = 200#3600*8#86400
num_paths = 1 

L = 1300     # real system width (e.g. 200 meters)
M = 2*L    # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate

# Source term:
λ = 1.0
μ = 0.1 
γ = 0.8

mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);
#seed = 3537480
seed = 3535956730

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,0.9,lag,true,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
RealStartTime = 6 # when, in real time, to kick the system
SimStartTime = to_simulation_time(RealStartTime,Δt)-2 # convert to simulation time
Position = 0
Volume = 10
# Volume set below



myRLPusher = RLPushTerm(SimStartTime,SimStartTime+2,Position,Volume,false,false)
kernel_cut = 0.00008

lob_model_1 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    do_exp_dist_times=false,do_test=false,
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);
lob_model_2 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    do_exp_dist_times=true,do_test=false,
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);
lob_model_3 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=false,
    do_exp_dist_times=true,do_test=true, #means to use uniformly distributed times
    Δx=target_Δx,seed=seed,kernel_cut_off=kernel_cut);

seeds = repeat([seed],3)

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_2.SK_DP

# +
# lob_model_1_temp = SLOB(num_paths, 1000, p₀, Δt, L, D, ν, α, γ, 
#     mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; store_past_densities=true,
#     do_exp_dist_times=true,do_test=false,
#     Δx=target_Δx,seed=seed);

# my_x_scale = 3.0
# my_y_scale = 0.3
# (Dat,arr1) = quick_plot([lob_model_1_temp],100,12;seed=seed,for_visual=(
#         #kw_for_plot=(ylim=[-my_y_scale,my_y_scale],),
#         x_axis_width=my_x_scale,))
# plot!()

# +
# plot(get_sums(Dat1[1][1].lob_densities;absolute=false))

# +
# my_file_name = "SF_DT"

# my_print_labels = ["\nDo run without any exponential stuff\n",
#              "\nDo run with exponentially distributed times\n",
#              "\nDo run with uniformally distributed times\n"]

# seeds = repeat([seed],3)
    
# (Dat2,Dat1,Dat3) = obtain_data_list([lob_model_2,lob_model_1,lob_model_3]
#                                         ;seeds=seeds,do_new=false,print_labels=my_print_labels,save_new=true,
#                                          folder_name="ReportData",file_name=my_file_name,
#                                          folder_path=global_folder_path);

# +
my_file_name = "SF_DD_2"

(Dat1,) = obtain_data_list([lob_model_1]
                            ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                             folder_name="ReportData",file_name=my_file_name,
                             folder_path=global_folder_path);

# +
my_file_name = "SF_DT_2"

(Dat2,) = obtain_data_list([lob_model_2]
                            ;seeds=seeds,do_new=true,print_labels=[""],save_new=false,
                             folder_name="ReportData",file_name=my_file_name,
                             folder_path=global_folder_path);

# +
# my_file_name = "SF_DT_1"

# (Dat3,) = obtain_data_list([lob_model_3]
#                             ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
#                              folder_name="ReportData",file_name=my_file_name,
#                              folder_path=global_folder_path);
# -

t_ = Dat2[1][1].slob.Δt
t_i = floor(Int64,1/t_)
temp1 = Dat2[1][1].obs_price_paths
temp2 = Dat2[1][1].raw_price_paths[1:t_i:end];

Dat2[1][1].obs_price_paths = temp2;

Dat2[1][1].slob.Δt      = Dat1[1][1].slob.Δt ;
Dat2[1][1].slob.Δts_cum = Dat1[1][1].slob.Δts_cum;

# +
sf = StylizedFactsPlot([Dat2,Dat1];do_raw=false)#,real_upper_cut_off=25000);

plot_all_stylized_facts(sf,plot_size=(2000,1000);tolerance=1.0)
# -



Dat2[1][1].slob.Δt

# +
my_file_name = "SF_DT"
folder_name = string(picture_folder_name,"/",my_file_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/3)

save_seperate(sf;folder_name=folder_name,file_name=my_file_name,folder_path=global_folder_path,save_type="png",
    scale_=1.0,plot_size=size_,dpi=dpi_,tolerance=[-0.22,1.0],
    for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0), zoom=2.01))
# -

mydata = [sf.log_returns[i,:] for i in 1:sf.L]

mrlplot(mydata[1])

using Extremes

# ## Most generic run

# +
# Configuration Arguments
# ran in 4.5 hours

T = 3600*8
num_paths = 1 

L = 1000     # real system width (e.g. 200 meters)
M = 2*L    # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate

# Source term:
λ = 1.0
μ = 0.1 
γ = 0.8

mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);

seed = NaN
#seed = 3535956730 #0
#seed = 4898384128 #1
#seed = 5554355463 #2
#seed = 0586258657 #3
#seed = 3453348462 #4
#seed = 4682022781 #5
#seed = 2491246639 #6

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm_1 = RandomnessTerm(σ,r,0.9,lag,true,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)


myRLPusher = RLPushTerm(0.0,0.0,0.0,0.0,false,false)
kernel_cut = 0.00008

lob_model_1 = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm_1; store_past_densities=false, 
    Δx =  target_Δx, seed=seed, kernel_cut_off = kernel_cut);

seeds = repeat([seed],3)

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_1.SK_DP

# +
my_file_name = NaN
#my_file_name = "SF_1"
#my_file_name = "SF_2"
#my_file_name = "SF_3"
#my_file_name = "SF_4"
#my_file_name = "SF_5"
#my_file_name = "SF_6"
    
(Dat1,) = obtain_data_list([lob_model_1]
                                         ;seeds=seeds,do_new=true,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);
# -



# +
sf = StylizedFactsPlot([Dat2,Dat1,Dat3];do_raw=false);

plot_all_stylized_facts(sf,plot_size=(2000,1000);tolerance=1.0,
for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=0.5))

# +
my_file_name = "SF_DC"
folder_name = string(picture_folder_name,"/",my_file_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=1/3)

save_seperate(sf;folder_name=folder_name,file_name=my_file_name,folder_path=global_folder_path,save_type="png",
    scale_=1.0,plot_size=size_,dpi=dpi_,tolerance=2.0,quick=false,
    for_mid = (legend_pos=(:left,:top),frac_choice1=(1,1),frac_choice2=(0,0),create_space=0.5))
# -

# ## Analyse data

# ### Get \sigma_h

# +
my_file_name = NaN
my_file_name = "SF_1"
#my_file_name = "SF_2"
#my_file_name = "SF_3"
seeds = []

Dats = Array{Dict{Int64, Dict{Int64, DataPasser}}}(undef,5)
    
(Dats[1],) = obtain_data_list([]
                                         ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

my_file_name = "SF_2"

(Dats[2],) = obtain_data_list([]
                                         ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

my_file_name = "SF_DD_2"

(Dats[3],) = obtain_data_list([]
                                         ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);


my_file_name = "SF_3"
(Dats[4],) = obtain_data_list([]
                                         ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);


my_file_name = "SF_4"
(Dats[5],) = obtain_data_list([]
                                         ;seeds=seeds,do_new=false,print_labels=[""],save_new=true,
                                          folder_name=dat_folder_name,file_name=my_file_name,
                                          folder_path=global_folder_path);

# +
divide_into = 8
changes = zeros(Float64,length(Dats) * divide_into)

count = 1
for dat in Dats
    price_path = dat[1][1].raw_price_paths
    gap = floor(Int64,length(price_path)/8)
    
    for i in 0:(divide_into-1)
        changes[count] = price_path[(i+1)*gap+1]-price_path[(i)*gap+1]
        count += 1
    end
end
# -

myfit = Distributions.fit(Normal,changes)
plot(myfit,label=L"Normal fit: $\mu=%$(round(myfit.μ,digits=2))$, $\sigma=%$(round(myfit.σ,digits=2))$ ",color=1)
scatter!(changes,repeat([0],length(changes)),label="Scatter visual",color=2)
histogram!(changes,alpha=0.2,normalize=true,bins=floor(Int64,length(changes)/2),label="Histogram visual",color=3)
vline!([myfit.μ-myfit.σ,myfit.μ,myfit.μ+myfit.σ],label="Mean+Std.dev",color=4)
plot!(xlab=L"$\Delta p$ over one hour",ylab="Counts")

σ_hourly = myfit.σ

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "CalculationOfHourlyVariance"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.4)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# ### Get \V_h

# +
# Configuration Arguments
plot()
L = 60
result = zeros(Float64,20)

r = 0.5
num_paths = 1#50#30

L = 60     # real system width (e.g. 200 meters)
M = 2*L     # divided into M pieces , 400

p₀ = 1300.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 0.8 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS) 

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
myCouplingTerm = CouplingTerm(μ, 0.0, 0.0, 0.0, false);

#seed = 3535956730
#seeds = repeat([seed],3)

# My randomness term
σ = 1.0 #variance in randomness
β = 0.9 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
kernel_cut = 0.00008

# RL Stuff:
T = 100
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

Dat = InteractOrderBooks([lob_model_push,lob_model_no_push], -1, true) ;

# +
l_,r_ = get_effective_market_orders(Dat;shift=0)
l_ = .-l_
derivs = get_central_derivative(Dat)

correc = (1-r/2) / (1-r) * (1/(-0.5(γ-1)+0.5)-1) #* (2-γ)/(1+0.5*γ)*γ*γ*γ
x = r_ .+ l_
y =  D.*abs.(derivs)*Δt

cumx = cumsum(x)
cumy = cumsum(y)

len = length(cumx[2:end-1])
xrange = (1:len)./len*100
rat = cumx ./ cumy

# +
p = plot()

p = scatter!(p,xrange,y[2:end-1],
    label=L"V_{D}^1(t)",
    markerstrokewidth=0.0,ms=0.6)
p = scatter!(p,xrange,x[2:end-1].*correc,
    label=L"V_{D}^2(t)",
    markerstrokewidth=0.0,ms=0.3)
#p = scatter!(p,xrange,y[2:end-1].*correc,
#    label=L"V_{D}^2(t)",
#    markerstrokewidth=0.0,ms=0.6)
p = plot!(p,legend=:bottomleft)
p = plot!(p,xlab = L"Time ( $t$ )", ylab = L"Volume traded by method $V_D^i(t)$")

frac = 0.45
p = plot!(p,inset = (1,bbox(0.05, 0.05, frac, frac, :bottom, :right))  )

p = plot!(p,xrange,cumy[2:end-1],subplot=2,
    label=L"\int_0^t dt' V_D^1(t')",lw=2,alpha=0.8)
p = plot!(p,xrange,cumx[2:end-1].*correc,subplot=2,
    label=L"\int_0^t dt' V_D^2(t')",lw=2,alpha=0.8)
#p = plot!(p,xrange,cumy[2:end-1].*correc,subplot=2,
#    label=L"\int_0^t dt' V_D^3(t')",lw=2,alpha=0.8)
#p = plot!([],label=L"",alpha=0,subplot=2)
p = plot!(p,subplot=2,legend=:topleft,frame=:box,xmirror=true)

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "CalculationOfHourlyVolume"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

myderiv = D.*abs.(derivs)[end]
V_hourly = myderiv * 25000 / 8 
V_daily = V_hourly * 8

# ### Final calc

# Y σ_D / (V_D)^0.8 = myfit
rhs = measurement(1.33701,0.00028)
lhs = σ_hourly*sqrt(8)/(V_daily)^0.8
Y = rhs/lhs

σ_hourly * sqrt(8)

myderiv 

0.0957*25000

# ## (NW) Different Delta x

# # (NW) MICHAELS VS NEW

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

#ν = 3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

ν = 0.1#1.0#3.0 #removal rate
ν = 1.0#3.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, true);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

Δx = L / M  # real gap between simulation points 
Δt = (r * (Δx^2) / (2.0 * D))^(1/γ)

# RL Stuff:
T = 20
RealKickStartTime = 8 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -2 
Volume = 10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false)

lob_model_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm,shift=-1,old_way=true);

lob_model_no_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm,shift=-1,old_way=true);

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model_push.SK_DP
# -

# check something actually happens for one example
if true
    myCouplingTerm = CouplingTerm(μ, a, b, c, false);
    
    Volume = 10

    myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true)
    myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false)
    
    lob_model_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm,shift=-1,michaels_way=true);
    
    lob_model_no_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm,shift=-1,michaels_way=true);
    
    Dat = quick_plot([lob_model_push,lob_model_no_push],SimKickStartTime,12)
    
    #png("/home/derickdiana/Desktop/Masters/Reworked/KickTheSytemWithRemoval.png")
    #png("/home/derickdiana/Desktop/Masters/Reworked/NumericalInstabilityForLargeRemovalRate.png")
    plot!()
end


plot_sums(Dat)

# +
gammas = [1.0]#[1.0,0.9,0.8,0.7,0.6]#[1.0,0.9,0.8,0.7,0.6]
volumes = range(1,100,length=100)

function get_set_inner(volume,γ,do_michaels_way,ν)
    Δt = (r * (Δx^2) / (2.0 * D))^(1/γ)
        
    l = Int(round(to_simulation_time(1,Δt)/3,digits=0))

    RealKickStartTime = 8 # when, in real time, to kick the system
    SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time

    #myCouplingTerm = CouplingTerm(μ, a, b, c, false);

    myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,volume,true)
    myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,volume,false)

    lob_model_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm, michaels_way = do_michaels_way);

    lob_model_no_push = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm, michaels_way = do_michaels_way);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
    
end

function get_set(volume,γ)
    return get_set_inner(volume,γ,true,2.0)
end
(mean_price_impacts_frac_no_random_michaels_way,var_price_impacts_frac_no_random_michaels_way) = calculate_price_impacts(volumes,gammas,get_set)

function get_set(volume,γ)
    return get_set_inner(volume,γ,false,2.0)
end
(mean_price_impacts_frac_no_random_new_way,var_price_impacts_frac_no_random_new_way) = calculate_price_impacts(volumes,gammas,get_set)

function get_set(volume,γ)
    return get_set_inner(volume,γ,true,2.0*6/10)
end
(mean_price_impacts_frac_no_random_nu_adjust_michaels_way,var_price_impacts_frac_no_random_nu_adjust_michaels_way) = calculate_price_impacts(volumes,gammas,get_set);


# +
for Gamma in 1:(gi_len)
    fit_and_plot_price_impact(volumes,mean_price_impacts_frac_no_random_michaels_way,var_price_impacts_frac_no_random_michaels_way,["Michaels with nu=2"];colors=["blue"],new_plot=true)
    fit_and_plot_price_impact(volumes,mean_price_impacts_frac_no_random_new_way,var_price_impacts_frac_no_random_new_way,["New way with nu=2"];colors=["red"])
    fit_and_plot_price_impact(volumes,mean_price_impacts_frac_no_random_nu_adjust_michaels_way,var_price_impacts_frac_no_random_nu_adjust_michaels_way,["Michaels moved closer to new way by setting nu = 1.3"];colors=["green"])
end
plot!()

#png("/home/derickdiana/Desktop/Masters/Reworked/PriceImpactFractionalAdaptiveDelayOnlyNoRandom.png")
#png("/home/derickdiana/Desktop/Masters/Reworked/PriceImpactMichaelsCode.png")
#png("/home/derickdiana/Desktop/Masters/Reworked/75ScaleWithRandom.png")
#png("/home/derickdiana/Desktop/Masters/Reworked/ThreeWays.png")
# -
# # NO RANDOM KICKS PRICE IMPACTS

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 80
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -2#-3 
Volume = 10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false,beginning_shift_frac=0.0 )#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false,beginning_shift_frac=0.0)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,parr) = quick_plot([lob_model_push],[],SimKickStartTime;for_visual=(
        x_axis_width=3,do_left=false,do_right=false)
)
plot!()

areas = get_sums(Dat[1][1].lob_densities;absolute=true)
plot(areas)
vline!([SimKickStartTime-2])

area = areas[SimKickStartTime-2] * target_Δx / 2 #area of one half

# +
lob = Dat[1][1].lob_densities[:,SimKickStartTime-1]
l = length(lob)
middle = floor(Int64,l/2) + 1
width = 25
plot(lob[(middle-width):(middle+width)])
vline!([width+1])
hline!([0];color="black")

sums = calculate_trapezium_area_many(lob,target_Δx,middle-20,middle)
sums

tick_volumes = cumsum(sums).*target_Δx
# -

mynorm = (V) -> V./(area)
mylog = (V) -> log.((V./area).+1)
mysqrt = (V) -> (V/area).^(0.5)

# +
# function draw_square_abs(plt,one_side,other_side)
#     plt = plot!(plt,[one_side[1]  ,other_side[1]],[one_side[2]  ,one_side[2]  ],color="black",label="")
#     plt = plot!(plt,[one_side[1]  ,other_side[1]],[other_side[2],other_side[2]],color="black",label="")
#     plt = plot!(plt,[one_side[1]  ,one_side[1]  ],[one_side[2],other_side[2]],color="black",label="")
#     plt = plot!(plt,[other_side[1],other_side[1]],[one_side[2],other_side[2]],color="black",label="")
    
#     return plt
# end
# -

# ## Different delays

# ### Limit order

# #### Price impact

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, 1.0,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, 1.0,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD"

volumes = exp.(range(-8,log(area^6),length=500))
sub = 1:length(volumes)
x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx


my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(
        p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=vcat(repeat([false],6),true),do_log_plot=false,
        do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,
        straight_line_frac_to_plot=[1.2,1.1,1.4,1.0,1.3,1.2,1.1],
        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false, 
        subplot_index=subplot_ind,sub = curr_sub
    )


frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)


# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with uncertainty

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    num_paths = 10
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)
    
    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    σ = 1.0 #variance in randomness
    β = 0.0 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = false #behave like a random walk
    
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, 1.0,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, 1.0,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD-uncert"

volumes = exp.(range(-8,log(area^6),length=500))
sub = 1:length(volumes)
x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx


my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(
        p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=vcat(repeat([false],6),true),do_log_plot=false,
        do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,
        straight_line_frac_to_plot=[1.2,1.1,1.4,1.0,1.3,1.2,1.1],
        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false, 
        subplot_index=subplot_ind,sub = curr_sub
    )


frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)
# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with spline

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD-Int"

# looks kinda straight in that range
#volumes = exp.(range(-4,1,length=1000))
# curve starts in this range
#volumes = exp.(range(-4,4,length=1000))

# can clearly see the curves
volumes = exp.(range(-8,log(area^6),length=500))
sub = 1:length(volumes)

x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx

my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_plot=false,
    do_vert_log_shift=true,do_horiz_log_shift=true,
    do_power_fit=false,do_log_fit=false,
    use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,
    straight_line_frac_to_plot=[1.3,1.3,1.4,1.5,1.6,1.7,1.7],
    modify_input_v=(v)->v/area, 
    modify_plot_v=(v)->log(v/area+1),
    modify_input_p=(p)->p,
    modify_plot_p=(p)->p/target_Δx,
    fit_labels_in_legend=false,subplot_index=subplot_ind,sub=curr_sub)#, do_just=[false,false,false,false,false,false,true])


frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)
# -

# #### Price impact with exp times

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD_Exp"

# looks kinda straight in that range
#volumes = exp.(range(-4,1,length=1000))
# curve starts in this range
#volumes = exp.(range(-4,4,length=1000))

# can clearly see the curves
volumes = exp.(range(-8,10,length=1000))
sub = 1:length(volumes)

x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = false, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx

my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=false,do_log_plot=false,do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,straight_line_frac_to_plot=[1.3,1.4,1.3,1.2,1.1,1.5,1.5],

        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false,
    subplot_index=subplot_ind,sub=curr_sub,
    tiny_shift = 0.05, do_just = [true,true,false,true,false,true,false])

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with exp times and interp

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD_Exp-Int"

# looks kinda straight in that range
#volumes = exp.(range(-4,1,length=1000))
# curve starts in this range
#volumes = exp.(range(-4,4,length=1000))

# can clearly see the curves
volumes = exp.(range(-8,10,length=1000))
sub = 1:length(volumes)

x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = false, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx

my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=false,do_log_plot=false,do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,straight_line_frac_to_plot=[1.3,1.4,1.3,1.2,1.1,1.5,1.5],

        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false,
    subplot_index=subplot_ind,sub=curr_sub,
    tiny_shift = 0.05, do_just = [true,true,false,true,false,true,false])

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"\mathcal{I}(Q,\Delta n)=(p(n+\Delta n)|_Q-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with exp times and interp and uncertainty

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    num_paths = 10
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    σ = 1.0 #variance in randomness
    β = 0.0 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = false #behave like a random walk
    
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)
    
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD_Exp-Int-uncert"

# looks kinda straight in that range
#volumes = exp.(range(-4,1,length=1000))
# curve starts in this range
#volumes = exp.(range(-4,4,length=1000))

# can clearly see the curves
volumes = exp.(range(-8,10,length=1000))
sub = 1:length(volumes)

x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx

my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=false,do_log_plot=false,do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,straight_line_frac_to_plot=[1.3,1.4,1.3,1.2,1.1,1.5,1.5],

        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false,
    subplot_index=subplot_ind,sub=curr_sub,
    tiny_shift = 0.05, do_just = [true,true,false,true,false,true,false])

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"\mathcal{I}(Q,\Delta n)=(p(n+\Delta n)|_Q-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.4)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with exp times and uncertainty

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    num_paths = 10
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    σ = 1.0 #variance in randomness
    β = 0.0 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = false #behave like a random walk
    
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)
    
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DD_Exp-uncert"

# looks kinda straight in that range
#volumes = exp.(range(-4,1,length=1000))
# curve starts in this range
#volumes = exp.(range(-4,4,length=1000))

# can clearly see the curves
volumes = exp.(range(-8,10,length=1000))
sub = 1:length(volumes)

x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
steps = 1:7
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = true, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx

my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=false,do_log_plot=false,do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,straight_line_frac_to_plot=[1.3,1.4,1.3,1.2,1.1,1.5,1.5],

        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false,
    subplot_index=subplot_ind,sub=curr_sub,
    tiny_shift = 0.05, do_just = [true,true,false,true,false,true,false])

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\ln(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# ### Market order

# #### Price impact 

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"(p(n+\Delta n)-p(n))/(\Delta x/2)"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO-general"
#my_file_name = "PI_DD-MO-log"
#my_file_name = "PI_DD-MO-log"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"l=%$(l[1])",combined)

upper = 0.13
x_inset_pos = findfirst((v)->mysqrt(v)>upper,volumes)
x_inset_val = mysqrt(volumes[x_inset_pos])
sub_ins = 1:x_inset_pos
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(0.0,mysqrt(2.0*tick_volumes))
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))
print(my_xticks_ins)

for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )

do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,#do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v/area,
    modify_plot_v  = (v)->mysqrt(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=do_these,type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),-1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"(V/A)^{0.5}",ylabel=L"(p(\ell_0+\ell)-p(\ell_0))/(\Delta x/2)"))

xb = 0.07
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,xticks=my_xticks_ins,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = plot!(p,xlims=[-0.005,x_inset_val],subplot=2)

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac-0.009,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb-0.013,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO-sqrt"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"l=%$(l[1])",combined)

my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = 2.0.*tick_volumes/area
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))

for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35
            )

do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v, # making this v makes the answer log
    modify_plot_v  = (v)->v/area,
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=do_these,type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),-1,sub,
    (for_plot_base...,legend=:topleft,
        xlabel=L"\log(V/A+1)",ylabel=L"(p(t+l)-p(t))/\Delta x"))

p = plot!(p, inset = (1,bbox(0.05, 0.05, frac, frac, :bottom, :right))  )

temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel=""))

# +
my_file_name = "PI_DD-MO-normal"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
##### Not yet working
my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)


my_yticks = vcat(0,range(0.25,5,step=0.25))
xticks_vals = 2.0*(tick_volumes./exp(upper))
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))

do_these = [true,true,true,false,false,false,true]
fit_and_plot_price_impact((steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    new_plot=true,
    forplot=(legend=:bottomright,yticks=my_yticks,size=my_size,
        gridalpha=0.35,xlabel="V/A",xticks=my_xticks),
    do_log_fit=do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    modify_input_v = (v)->-log(1-v/exp(upper)), 
    modify_plot_v  = (v)->v/exp(upper),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p,
   
    do_just=do_these)
# -

# #### Price impact with interpolation

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO-Int"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"(p(n+\Delta n)-p(n))/(\Delta x/2)"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
folder_name = string(picture_folder_name,"/","Singles")

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)
# -

# #### Price impact with uncertainty

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    r = 0.5 #proportion of time in which it jumps left or right
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    num_paths = 10
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    σ = 1.0 #variance in randomness
    β = 0.0 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = false #behave like a random walk
    
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO-uncert"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = false, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"(p(n+\Delta n)-p(n))/(\Delta x/2)"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO-general-uncert"
#my_file_name = "PI_DD-MO-log"
#my_file_name = "PI_DD-MO-log"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)
# -

# #### Price impact with exp

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO_exp"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"(p(n+\Delta n)-p(n))/(\Delta x/2)"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO_exp-general"
#my_file_name = "PI_DD-MO-log"
#my_file_name = "PI_DD-MO-log"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"l=%$(l[1])",combined)

upper = 0.13
x_inset_pos = findfirst((v)->mysqrt(v)>upper,volumes)
x_inset_val = mysqrt(volumes[x_inset_pos])
sub_ins = 1:x_inset_pos
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(0.0,mysqrt(2.0*tick_volumes))
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))
print(my_xticks_ins)

for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )

do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,#do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v/area,
    modify_plot_v  = (v)->mysqrt(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=do_these,type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),-1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"(V/A)^{0.5}",ylabel=L"(p(\ell_0+\ell)-p(\ell_0))/(\Delta x/2)"))

xb = 0.07
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,xticks=my_xticks_ins,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = plot!(p,xlims=[-0.005,x_inset_val],subplot=2)

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac-0.009,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb-0.013,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO-sqrt"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"l=%$(l[1])",combined)

my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = 2.0.*tick_volumes/area
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))

for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35
            )

do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v, # making this v makes the answer log
    modify_plot_v  = (v)->v/area,
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=do_these,type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),-1,sub,
    (for_plot_base...,legend=:topleft,
        xlabel=L"\log(V/A+1)",ylabel=L"(p(t+l)-p(t))/\Delta x"))

p = plot!(p, inset = (1,bbox(0.05, 0.05, frac, frac, :bottom, :right))  )

temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel=""))

# +
my_file_name = "PI_DD-MO-normal"
folder_name = string(picture_folder_name,"/","Singles")


save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)

# +
##### Not yet working
my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)


my_yticks = vcat(0,range(0.25,5,step=0.25))
xticks_vals = 2.0*(tick_volumes./exp(upper))
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))

do_these = [true,true,true,false,false,false,true]
fit_and_plot_price_impact((steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    new_plot=true,
    forplot=(legend=:bottomright,yticks=my_yticks,size=my_size,
        gridalpha=0.35,xlabel="V/A",xticks=my_xticks),
    do_log_fit=do_these,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    modify_input_v = (v)->-log(1-v/exp(upper)), 
    modify_plot_v  = (v)->v/exp(upper),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p,
   
    do_just=do_these)
# -

# #### Price impact with exp non-uniform

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO_exp-Int-general"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"\mathcal{I}(Q,\Delta n)=(p(n+\Delta n)|_Q-p(n))/\Delta x"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
my_file_name = "PI_DD-MO_exp-Int-general"
#my_file_name = "PI_DD-MO-log"
#my_file_name = "PI_DD-MO-log"
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)
size_ = size_.*0.7

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# #### Price impact with exp non-uniform with uncert

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,l)
end

function get_set_inner(volume,l)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    num_paths = 3#20
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    σ = 1.0 #variance in randomness
    β = 0.0 #probability of being the value of the previous lag or mean reversion strength
    lag = 10 #lag
    do_random_walk = false #behave like a random walk
    
    myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=true,Δx=target_Δx,do_exp_dist_times=true, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO_exp-Int-general-uncert"

# looks kinda straight in that range
#upper = log(0.99 * λ/(2*μ)/2)

#volumes = exp.(range(-8,log(0.3*area),length=600))
volumes = range(0.0001,0.3*area,length=1000)
steps = 1:7
sub = 1:length(volumes)

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = false, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
#pgfplotsx()
# -

gr()

# +
#############################################################

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=my_size,gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"\mathcal{I}(Q,\Delta n)=(p(n+\Delta n)|_Q-p(n))/\Delta x"))

# +
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
#############################################################
# -

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.4)
size_ = size_ #.* pm.pt

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)

my_labels = map(l -> L"\Delta n=%$(l[1])",combined)

myfunc = (v) -> log.(v.+1)
myfunc = (v) -> sqrt.(v)
myfunc = (v) -> (v)

upper = 0.15
x_inset_pos = findfirst((v)->myfunc(v)>upper,volumes)
x_inset_val = myfunc(volumes[x_inset_pos])
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/(target_Δx/2)

sub_ins = 1:x_inset_pos


my_yticks = vcat(0,range(0.0,50,step=1.0))
xticks_vals = vcat(myfunc(2.0.*tick_volumes),0.0)
my_xticks = (xticks_vals, ((l)->round(l,digits=2)).(xticks_vals))
my_xticks_ins = (xticks_vals[xticks_vals.<upper], 
            ((l)->round(l,digits=2)).(xticks_vals[xticks_vals.<upper]))


for_plot_base =  
            (
            yticks=my_yticks,
            size=size_,
            dpi=dpi_,
            gridalpha=0.35
            )


do_these = [true,true,true,false,false,false,true]


temp(p,subplot_ind,curr_sub,for_plot) =  
fit_and_plot_price_impact(p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
    forplot=for_plot,
    do_log_fit=true,
    do_power_fit=do_these,
    do_log_plot=true,do_vert_log_shift=true,do_horiz_log_shift=true,
    use_only_straight_region=false,do_kinks=false,shift_y_ticks=true,
    
    # log x axis
    modify_input_v = (v)->v,
    modify_plot_v  = (v)->myfunc(v),
    modify_input_p = (p)->p,
    modify_plot_p  = (p)->p/(target_Δx/2),
    fit_labels_in_legend = false,
    do_just=[true,true,false,false,false,false,true],type_1_log=false,
    subplot_index=subplot_ind,sub=curr_sub)

frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,xticks=my_xticks,
        xlabel=L"Q",ylabel=L"\mathcal{I}(Q,\Delta n)=(p(n+\Delta n)|_Q-p(n))/\Delta x"))
# -

xb = 0.03
frac_ = floor(Int64,(size_[1] * frac)/pm.pct)*pm.pct
p = plot!(p, inset = (1,bbox(xb, xb, frac_, frac_, :top, :left))  )

# +
p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true,xticks=my_xticks_ins))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)

# +
#my_file_name = "PI_DD-MO-log"
#my_file_name = "PI_DD-MO-log"
folder_name = string(picture_folder_name,"/","Singles")

size_ = size_ 
dpi_ = dpi_ 

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# ## Different gammas

# +
function get_set(volume,combined_slice)
    l = combined_slice[1]
    return get_set_inner(volume,γ)
end

function get_set_inner(volume,γ)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,10,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+1, -2, volume,  true, false)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, false)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,
        do_exp_dist_times=false, do_test = false, 
        seed = seed,beginning_shift_frac=0.5);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime-1, l)
end


# +
my_file_name = "PI_DG"

volumes = exp.(range(-8,log(area^6),length=500))
sub = 1:length(volumes)
x_inset_pos = findfirst((v)->log(v/area+1)>1,volumes)
x_inset_val = log(volumes[x_inset_pos]/area+1)
sub_ins = 1:x_inset_pos

#what to try
gammas = 1.0,0.9
#all combinations of the above
combined = collect(Iterators.product(steps))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps,volumes,  combined), get_set; 
                                                        do_new = false, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,
                                                        folder_path=global_folder_path);
# -
function draw_square_abs(plt,one_side,other_side)
    plt = plot!(plt,[one_side[1]  ,other_side[1]],[one_side[2]  ,one_side[2]  ],color="black",label="")
    plt = plot!(plt,[one_side[1]  ,other_side[1]],[other_side[2],other_side[2]],color="black",label="")
    plt = plot!(plt,[one_side[1]  ,one_side[1]  ],[one_side[2],other_side[2]],color="black",label="")
    plt = plot!(plt,[other_side[1],other_side[1]],[one_side[2],other_side[2]],color="black",label="")
    
    return plt
end

# +
#my_labels = map(l -> string("Price impact after ",l[1]," time steps"),combined)
my_labels = map(l -> L"\Delta n=%$(l[1])",combined)
y_inset_val = maximum(mean_price_impacts[:,x_inset_pos,7])/target_Δx


my_yticks = vcat(0,range(0.5,10,step=1.0))
my_xticks = 0:20
for_plot_base =  
            (
            yticks=my_yticks,xticks=my_xticks,
            size=my_size,gridalpha=0.35,
            )

temp(p,subplot_ind,curr_sub,for_plot) =  
    fit_and_plot_price_impact(
        p,(steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
        forplot=for_plot,
        do_log_fit=vcat(repeat([false],6),true),do_log_plot=false,
        do_vert_log_shift=true,do_horiz_log_shift=true,do_power_fit=false,
        use_only_straight_region=true,do_kinks=false,shift_y_ticks=true,
        straight_line_frac_to_plot=[1.2,1.1,1.4,1.0,1.3,1.2,1.1],
        modify_input_v=(v)->v/area, 
        modify_plot_v=(v)->log(v/area+1),
        modify_input_p=(p)->p, 
        modify_plot_p=(p)->p/target_Δx, 
        fit_labels_in_legend=false, 
        subplot_index=subplot_ind,sub = curr_sub
    )


frac = 0.3

p = temp(plot(),1,sub,
    (for_plot_base...,legend=:bottomright,
        xlabel=L"\log(Q/A+1)",ylabel=L"(p(n+\Delta n)-p(n))/\Delta x"))

xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left))  )

p = temp(p,2,sub_ins,(for_plot_base...,legend=:none,
        xlabel="",ylabel="",frame=:box,xmirror=true,ymirror=true))

p = arrow_from_abs_to_frac(p,[x_inset_val,y_inset_val],[xb+frac,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[0,y_inset_val],[xb,1-(xb+frac)])
p = draw_square_abs(p,[0,0],[x_inset_val,y_inset_val])
plot!(p)
# +
folder_name = string(picture_folder_name,"/","Singles")

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

# # PRICE IMPACT PATH

# +
# Configuration Arguments
num_paths = 1#50#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.5 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

#ν = 1.0#1.0#3.0 #removal rate
#γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

# Source term:
λ = 1.0 #
μ = 0.18 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 0.01  #gap between stocks before at full strength: strong is 0.3
b = 2.0   #weighting of interaction term: strong is 2
c = 2.0   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# RL Stuff:
T = 40
RealKickStartTime = 16 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = -2#-3 
Volume = 10

myRLPusherPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
myRLPusherNoPush = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,false,false)

lob_model_push = SLOB(num_paths, T, p₀, Δt , L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false,beginning_shift_frac=0.0 )#michaels_way=true,do_interp=true);

lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush,myRandomnessTerm; Δx=target_Δx, do_exp_dist_times=false,beginning_shift_frac=0.0)#,michaels_way=true)#,do_interp=true);

lob_model_push.SK_DP
# -

(Dat,parr) = quick_plot([lob_model_push],[],SimKickStartTime;for_visual=(
        x_axis_width=3,do_left=false,do_right=false)
)
plot!()

# +
lob = Dat[1][1].lob_densities[:,SimKickStartTime-1]
l = length(lob)
middle = floor(Int64,l/2) + 1
width = 25
plot(lob[(middle-width):(middle+width)])
vline!([width+1])
hline!([0];color="black")

sums = calculate_trapezium_area_many(lob,target_Δx,middle-6,middle)
sums

tick_volumes = cumsum(sums).*target_Δx
# -

mynorm = (V) -> V./exp(upper)
mylog = (V) -> log.(mynorm(V).+1)
mysqrt = (V) -> V.^(0.5)

# ## Market order with different nu

# +
function get_set(volume,combined_slice)
    ν = combined_slice[1]
    return get_set_inner(volume,ν)
end

function get_set_inner(volume,ν)
    L = 200
    M = 400
    target_Δx = L/M
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,300,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+10, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DN-MO"

# looks kinda straight in that range
upper = log(0.99 * λ/(2*μ)/2)

volumes = exp(-1)#exp.(range(-8,upper,length=2000))
steps = 1:100
νs = [0.1,0.2,0.3]

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(νs))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
my_labels = map(ν_ -> string("Price impact with ν=",round(ν_[1];digits=2)),combined)

my_yticks = vcat(0,range(0.25,5,step=0.25))
xticks_vals = mylog(2.0.*tick_volumes)
my_xticks = (xticks_vals, ((ν)->round(ν,digits=2)).(xticks_vals))

fit_and_plot_price_change((steps,[1],my_labels),mean_price_impacts,var_price_impacts,kick_end_time=10,
                            modify_input_p = (t)->log(t))
plot!()

# +
folder_name = string(picture_folder_name,"/","Singles")

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)
# -

# ## Market order with different D

# +
function get_set(volume,combined_slice)
    ν = combined_slice[1]
    return get_set_inner(volume,D)
end

function get_set_inner(volume,D)
    L = 200
    M = 400
    target_Δx = L/M
    ν = 0.5
    Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)#(r * (target_Δx^2) / (2.0 * D))^(1/γ)
    seed = 67834673
    seed = 67822222
    
    (T,Δts,SimKickStartTime) = get_length_of_time_that_allows_kick(RealKickStartTime,300,Δt,seed)

    myRLPusherPush   = RLPushTerm(SimKickStartTime, SimKickStartTime+10, -2, volume,  false, true)
    myRLPusherNoPush = RLPushTerm(0.0, 0.0, 0.0, 0.0, false, true)
    
    lob_model_push    = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherPush,   myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);

    lob_model_no_push = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusherNoPush, myRandomnessTerm; 
        do_interp=false,Δx=target_Δx,do_exp_dist_times=false, do_test = false, seed = seed, beginning_shift_frac = 0.0);
    
    return ([lob_model_push, lob_model_no_push], SimKickStartTime, l)
end


# +
my_file_name = "PI_DD-MO"

# looks kinda straight in that range
upper = log(0.99 * λ/(2*μ)/2)

volumes = exp(-1)#exp.(range(-8,upper,length=2000))
steps = 1:100
Ds = [0.5,0.6,0.7]

#all combinations of the above. Usually its own thing
combined = collect(Iterators.product(Ds))[:]

(mean_price_impacts,var_price_impacts) = obtain_price_impacts((steps, volumes,  combined),  get_set; 
                                                        do_new = true, save_new = true, 
                                                        folder_name=dat_folder_name,file_name=my_file_name,folder_path=global_folder_path);
# +
my_labels = map(D_ -> string("Price impact with D=",round(D_[1];digits=2)),combined)

my_yticks = vcat(0,range(0.25,5,step=0.25))
xticks_vals = mylog(2.0.*tick_volumes)
my_xticks = (xticks_vals, ((ν)->round(ν,digits=2)).(xticks_vals))

fit_and_plot_price_change((steps,[1],my_labels),mean_price_impacts,var_price_impacts,kick_end_time=10)
plot!()

# +
folder_name = string(picture_folder_name,"/","Singles")

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=my_dpi,plot_size=my_size,notify=true)
# -

# # SINGLE POINT DIFFUSION

# +
# Configuration Arguments
num_paths = 1#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 1230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 0.0 #removal rate
γ = 0.8 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

mySourceTerm = SourceTerm(0.0, 0.0, false);

myCouplingTerm = CouplingTerm(0.0, 0.0, 0.0, 0.0, false);

# My randomness term
σ = 1.0#6.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.1 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = true #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

target_Δx = L / M  # real gap between simulation points 
Δt = calculate_Δt_from_Δx(target_Δx,γ,D,r)

# the below is used to get the cutoff
#T = 1000
#lob_model = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
#    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm);

# then we make the actual RLPushTerm
#RealKickStartTime = to_real_time(lob_model.cut_off,Δt) # when, in real time, to kick the system
#SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
#Position = Int(M/2)
#Volume = -80; # If position == -x where x>=0, then put it x above the mid price each time

T = 100

SimKickStartTime = length(calculate_modified_sibuya_kernel(γ,ν,T,Δt))*2
Position = Int(M/2)
Volume = -80; # If position == -x where x>=0, then put it x above the mid price each time
RealKickStartTime = to_real_time(SimKickStartTime,Δt)

myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,Volume,true,false)
kernel_cut = 0.0008

T = to_real_time(4*SimKickStartTime,Δt)+1
lob_model = SLOB(num_paths, T, p₀, Δt, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; shift=-1,Δx = target_Δx,kernel_cut_off=kernel_cut);

MinRealKickStartTime = RealKickStartTime# when, in real time, to kick the system. Starts late to give system time to relax. Must be latest of all the gammas to try
print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model.SK_DP

MinRealKickStartTime
# -

(Dat,p_arr) = quick_plot([lob_model],[],SimKickStartTime-1,12;
    for_visual=(do_left=false,do_right=false,x_axis_width=10,dosum=false,plot_raw_price=true))
plot!()

test = Dat[1][1].lob_densities[:,SimKickStartTime+5]
t_mean = mymean(test/sum(test) ,lob_model.x,lob_model.Δx;option=1)
t_var = myvar(test/sum(test) ,lob_model.x,lob_model.Δx;option=1)
print(t_mean," ",t_var^0.5)
plot(lob_model.x,test/sum(test),label="Data")
plot!(xlim=[t_mean-10,t_mean+10])
plot!([t_mean],seriestype="vline",color="red",label="mean")
plot!([t_mean - t_var^0.5],seriestype="vline",color="green",label="1 st.dev.")
plot!([t_mean + t_var^0.5],seriestype="vline",color="green",label="")

# +
function plot_gamma_spread(plt,variances,sums,all_gamma_indices,how_long,all_delta_ts;zoom_val=-1,label_fit=true,for_plot=(),subplot_index=-1)
    mr(f) = round(f,digits=5)
    if subplot_index > 0
        for_plot = (for_plot...,subplot=subplot_index)
    end

    for Gamma in 1:length(all_gamma_indices)
        
        curr_gamma = all_gamma_indices[Gamma]
        curr_dt = all_delta_ts[Gamma]

        l = to_simulation_time(how_long,curr_dt) 
        # how many sim time steps would have passed given the current \gamma
        # this tells you what subset to take of the total length vector is not just 0.0
        sub = 1:l-1 #take all those values
        
        # take as much as you can by default
        real_time = sub.*curr_dt
        curr_variances = variances[Gamma,sub]

        # but if you need to zoom, then subset again
        t_sub = sub
        if zoom_val > 0.0
            t_sub = real_time .< zoom_val
            t_real_time = real_time[sub]
            t_curr_variances = curr_variances[sub]
        end

        #theoretical value for b in b*x^gamma
        theoret_b = round((2*D)/gamma(1+curr_gamma),digits=2) 

        ((a,b),(au,bu)) = my_power_fit(real_time[t_sub],curr_variances[t_sub])
        theoretical_var = theoret_b.*real_time.^curr_gamma
        fitted_var = a.*real_time.^b


        scatter!(plt,
            real_time[t_sub],curr_variances[t_sub],
            label=string(L"\alpha=",curr_gamma),
            color=Gamma,markersize=1,markerstrokewidth=0.2;for_plot...)
        
        
        mf = (x) -> floor(Int64,x)
        mf2 = (x) -> !iszero(x)
        t_sub_e = diff(mf.(real_time[t_sub]))
        t_sub_e = vcat(1,t_sub_e[2:end],1)
        t_sub_e = mf2.(t_sub_e)
       
        
        scatter!(plt,
            real_time[t_sub][t_sub_e],curr_variances[t_sub][t_sub_e],
            color=Gamma,markersize=3,markerstrokewidth=0.2,label="";for_plot...)


        curr_label = ""
        if label_fit
            curr_label = L"Theoretical fit for $\alpha=%$curr_gamma$ is $y=%$theoret_b t^{%$curr_gamma}$ "
        end
        
        plot!(plt,
          real_time[t_sub],theoretical_var[t_sub],
          ls=:solid,color=Gamma,label=curr_label;for_plot...)

        f_unc_sa = f_unc(a,au)
        f_unc_sb = f_unc(b,bu)

       curr_label = ""
       if label_fit
           curr_label = string(L"Numeric fit for $\alpha=%$curr_gamma$ is y=",L"%$f_unc_sa","t^",L"%$f_unc_sb")
       end

       plot!(plt,
           real_time[t_sub],fitted_var[t_sub],
           ls=:dash,color=Gamma,label=curr_label;for_plot...)
       
       plot!(legendfontsize=17.0,legend=:bottomright)
    end


    folder_path = string(global_folder_path,"/",picture_folder_name)

    (dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

    plot!(plt,xlab="Real time (t)",ylab="Variance of spike";for_plot...)
    save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                    folder_path = folder_path, file_name = my_file_name,folder_name = "Singles", 
                    for_plot=(legendfontsize=12,),scale_=1.0)
    return plot!()
end

function plot_gamma_spread(variances,sums,all_gamma_indices,how_long,all_delta_ts;zoom_val=-1,label_fit=true,for_plot=())
    return plot_gamma_spread(plot(),variances,sums,all_gamma_indices,how_long,all_delta_ts;zoom_val=zoom_val,label_fit=label_fit,for_plot=for_plot)
end
# -

# ### Normal Dx

# +
# works with kernel_cut_off =0.000001
#gamma_indices = [1.0,0.9,0.8,0.7,0.6,0.5,0.4]
#ν = 0.0 #removal rate
#γ above set to 0.6
# ... up to the fact that the original thing is not linear

my_file_name = "VarSpikeData" # the one that takes long

L = 200
M = 400
target_Δx = L/M

gamma_indices = [1.0,0.9,0.8,0.7,0.6]#[1.0,0.9,0.8,0.7,0.6,0.5]#,0.9,0.8,0.7,0.6,0.5,0.4]
gi_len = length(gamma_indices)

delta_ts = map(γ->calculate_Δt_from_Δx(target_Δx,γ,D,r),gamma_indices)

how_long = 15

# determine the largest storage space that you will need
min_γ = 0.5
min_Δt = calculate_Δt_from_Δx(target_Δx,min_γ,D,r)
sim_longest = to_simulation_time(how_long,min_Δt)

# old way of determining the most space you will need
#min_γ = minimum(gamma_indices,dims=1)[1]
#min_Δt = minimum(delta_ts,dims=1)[1]
#sim_how_long = to_simulation_time(how_long,min_Δt)

MinSimKickStartTime = length(calculate_modified_sibuya_kernel(min_γ,ν,T,Δt))
MinRealKickStartTime = to_real_time(MinSimKickStartTime,min_Δt) + 1

variances = zeros(Float64,gi_len,sim_longest-1)
sums = zeros(Float64,gi_len,sim_longest-1);


# +
for Gamma in 1:gi_len
    γ = gamma_indices[Gamma] #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)
    Δt = delta_ts[Gamma]
    
    RealKickStartTime = 1#max(500,MinRealKickStartTime)
    SimKickStartTime = 2#to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time

    myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,-8,true,false)
    
    
    lob_model = SLOB(num_paths, RealKickStartTime+how_long+1, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; shift=-1 , kernel_cut_off=0.00008, Δx = target_Δx )
    
    print(lob_model.N," ",length(lob_model.SK_DP)," ",lob_model.N - length(lob_model.SK_DP),"\n")
    
    try clear_double_dict(Dat) catch e print("Not initialized") end
    GC.gc()

    Dat = InteractOrderBooks([lob_model], -1, true) 
    
    sim_how_long = to_simulation_time(how_long,delta_ts[Gamma])
    for t in 1:sim_how_long-1
        temp2 = Dat[1][1].lob_densities[:,SimKickStartTime+t] #across all x values, at time "t",  on path 1
        sums[Gamma,t] = sum(temp2)
        variances[Gamma,t] = myvar(temp2./sum(temp2),lob_model.x,lob_model.Δx;option=2)
    end
    
end


# +
l2 =  @layout [a;b]
plot()

all_gamma_indices = [1.0,0.9,0.8,0.7,0.6]
all_delta_ts = map(γ->calculate_Δt_from_Δx(target_Δx,γ,D,r),all_gamma_indices)
label_fit = false

p = plot_gamma_spread(variances,sums,all_gamma_indices,how_long,all_delta_ts;zoom_val=0.0,label_fit=true,for_plot=(legend=:none,))

frac = 0.3
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left)))
p = draw_square_abs(p,[0,0],[1,1])
p = arrow_from_abs_to_frac(p,[0,1],[xb,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[1,1],[xb+frac,1-(xb+frac)])
p = plot!(p, mirror=true,subplot=2, frame=:box  )

plot_gamma_spread(p,variances,sums,all_gamma_indices,how_long,all_delta_ts;
    zoom_val=1.0,label_fit=false,for_plot=(legend=:none,xlab="",ylab=""),subplot_index=2)
# -

plot!(legend=:none)

# +
file_name = "AnomDiff"
folder_path = string(global_folder_path,"/",picture_folder_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)

save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                folder_path = folder_path, file_name = file_name,folder_name = "Singles", 
                for_plot=(legendfontsize=12,),scale_=1.0)
plot!()
# -
# ### Small Dx

# #### Generate it all again

# +
my_file_name = "VarSpikeDataSmallDx" # the one that takes long

L = 80
M = 400
target_Δx = L/M

how_long = 20

# determine the largest storage space that you will need
min_γ = 0.5
min_Δt = calculate_Δt_from_Δx(target_Δx,min_γ,D,r)
sim_longest = to_simulation_time(how_long,min_Δt)

holder_variances = zeros(Float64,6,sim_longest-1)
holder_sums = zeros(Float64,6,sim_longest-1);

global_kernel_cut = 0.0000000001

# +
# works with kernel_cut_off =0.000001
#gamma_indices = [1.0,0.9,0.8,0.7,0.6,0.5,0.4]
#ν = 0.0 #removal rate
#γ above set to 0.6
# ... up to the fact that the original thing is not linear

gamma_indices = [1.0,0.9,0.8,0.7]
gi_len = length(gamma_indices)

variances = zeros(Float64,gi_len,sim_longest-1)
sums = zeros(Float64,gi_len,sim_longest-1);

delta_ts = map(γ->calculate_Δt_from_Δx(target_Δx,γ,D,r),gamma_indices)

# old way of determining the most space you will need
#min_γ = minimum(gamma_indices,dims=1)[1]
#min_Δt = minimum(delta_ts,dims=1)[1]
#sim_how_long = to_simulation_time(how_long,min_Δt)

MinSimKickStartTime = length(calculate_modified_sibuya_kernel(min_γ,ν,T,Δt))
MinRealKickStartTime = to_real_time(MinSimKickStartTime,min_Δt) + 1

for Gamma in 1:gi_len
    γ = gamma_indices[Gamma] #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)
    Δt = delta_ts[Gamma]
    
    RealKickStartTime = 1#max(500,MinRealKickStartTime)
    SimKickStartTime = 2#to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time

    myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,-8,true,false)
    
    
    lob_model = SLOB(num_paths, RealKickStartTime+how_long+1, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; shift=-1 , kernel_cut_off=global_kernel_cut, Δx = target_Δx )
    
    print(lob_model.N," ",length(lob_model.SK_DP)," ",lob_model.N - length(lob_model.SK_DP),"\n")
    
    try clear_double_dict(Dat) catch e print("Not initialized") end
    GC.gc()

    Dat = InteractOrderBooks([lob_model], -1, true) 
    
    sim_how_long = to_simulation_time(how_long,delta_ts[Gamma])
    for t in 1:sim_how_long-1
        temp = Dat[1][1].lob_densities[:,SimKickStartTime+t] #across all x values, at time "t",  on path 1
        sums[Gamma,t] = sum(temp)
        variances[Gamma,t] = myvar(temp./sum(temp),lob_model.x,lob_model.Δx;option=2)
    end
    
end


# +
holder_variances[1,:] = variances[1,:];
holder_sums[1,:] = sums[1,:];

holder_variances[2,:] = variances[2,:]
holder_sums[2,:] = sums[2,:];

holder_variances[3,:] = variances[3,:];
holder_sums[3,:] = sums[3,:];

holder_variances[4,:] = variances[4,:];
holder_sums[4,:] = sums[4,:];
# -

if target_Δx == 80/400
    save_data((holder_variances,holder_sums);
        file_name=my_file_name,folder_name=dat_folder_name,folder_path=global_folder_path)
end

# +
# works with kernel_cut_off =0.000001
#gamma_indices = [1.0,0.9,0.8,0.7,0.6,0.5,0.4]
#ν = 0.0 #removal rate
#γ above set to 0.6
# ... up to the fact that the original thing is not linear

gamma_indices = [0.6]
gi_len = length(gamma_indices)

variances = zeros(Float64,gi_len,sim_longest-1)
sums = zeros(Float64,gi_len,sim_longest-1);

delta_ts = map(γ->calculate_Δt_from_Δx(target_Δx,γ,D,r),gamma_indices)

# old way of determining the most space you will need
#min_γ = minimum(gamma_indices,dims=1)[1]
#min_Δt = minimum(delta_ts,dims=1)[1]
#sim_how_long = to_simulation_time(how_long,min_Δt)

MinSimKickStartTime = length(calculate_modified_sibuya_kernel(min_γ,ν,T,Δt))
MinRealKickStartTime = to_real_time(MinSimKickStartTime,min_Δt) + 1

for Gamma in 1:gi_len
    γ = gamma_indices[Gamma] #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)
    Δt = delta_ts[Gamma]
    
    RealKickStartTime = 1#max(500,MinRealKickStartTime)
    SimKickStartTime = 2#to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time

    myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,-8,true,false)
    
    
    lob_model = SLOB(num_paths, RealKickStartTime+how_long+1, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; shift=-1 , kernel_cut_off=global_kernel_cut, Δx = target_Δx )
    
    print(lob_model.N," ",length(lob_model.SK_DP)," ",lob_model.N - length(lob_model.SK_DP),"\n")
    
    try clear_double_dict(Dat) catch e print("Not initialized") end
    GC.gc()

    Dat = InteractOrderBooks([lob_model], -1, true) 
    
    sim_how_long = to_simulation_time(how_long,delta_ts[Gamma])
    for t in 1:sim_how_long-1
        temp = Dat[1][1].lob_densities[:,SimKickStartTime+t] #across all x values, at time "t",  on path 1
        sums[Gamma,t] = sum(temp)
        variances[Gamma,t] = myvar(temp./sum(temp),lob_model.x,lob_model.Δx;option=2)
    end
    
end
# -


holder_variances[5,:] = variances[1,:]
holder_sums[5,:] = sums[1,:]

holder_variances

if target_Δx == 80/400
    save_data((holder_variances,holder_sums);
        file_name=my_file_name,folder_name=dat_folder_name,folder_path=global_folder_path)
end

# +
# works with kernel_cut_off =0.000001
#gamma_indices = [1.0,0.9,0.8,0.7,0.6,0.5,0.4]
#ν = 0.0 #removal rate
#γ above set to 0.6
# ... up to the fact that the original thing is not linear

gamma_indices = [0.5]
gi_len = length(gamma_indices)

variances = zeros(Float64,gi_len,sim_longest-1)
sums = zeros(Float64,gi_len,sim_longest-1);

delta_ts = map(γ->calculate_Δt_from_Δx(target_Δx,γ,D,r),gamma_indices)

# old way of determining the most space you will need
#min_γ = minimum(gamma_indices,dims=1)[1]
#min_Δt = minimum(delta_ts,dims=1)[1]
#sim_how_long = to_simulation_time(how_long,min_Δt)

MinSimKickStartTime = length(calculate_modified_sibuya_kernel(min_γ,ν,T,Δt))
MinRealKickStartTime = to_real_time(MinSimKickStartTime,min_Δt) + 1

for Gamma in 1:gi_len
    γ = gamma_indices[Gamma] #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)
    Δt = delta_ts[Gamma]
    
    RealKickStartTime = 1#max(500,MinRealKickStartTime)
    SimKickStartTime = 2#to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time

    myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+1,Position,-8,true,false)
    
    
    lob_model = SLOB(num_paths, RealKickStartTime+how_long+1, p₀, Δt, L, D, ν, α, γ,
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm; shift=-1 , kernel_cut_off=global_kernel_cut, Δx = target_Δx )
    
    print(lob_model.N," ",length(lob_model.SK_DP)," ",lob_model.N - length(lob_model.SK_DP),"\n")
    
    try clear_double_dict(Dat) catch e print("Not initialized") end
    GC.gc()

    Dat = InteractOrderBooks([lob_model], -1, true) 
    
    sim_how_long = to_simulation_time(how_long,delta_ts[Gamma])
    for t in 1:sim_how_long-1
        temp = Dat[1][1].lob_densities[:,SimKickStartTime+t] #across all x values, at time "t",  on path 1
        sums[Gamma,t] = sum(temp)
        variances[Gamma,t] = myvar(temp./sum(temp),lob_model.x,lob_model.Δx;option=2)
    end
    
end
# -


holder_variances[6,:] = variances[1,:]
holder_sums[6,:] = sums[1,:]

if target_Δx == 80/400
    save_data((holder_variances,holder_sums);
        file_name=my_file_name,folder_name=dat_folder_name,folder_path=global_folder_path)
end

# #### Load it and plot it

# loads the one that takes long
my_file_name = "VarSpikeDataSmallDx" # the one that takes long
(holder_variances,holder_sums) = load_data(dat_folder_name,my_file_name;folder_path=global_folder_path)

# +
variances = holder_variances
sums = holder_sums
all_gamma_indices = [1.0,0.9,0.8,0.7,0.6]
how_long = 15
all_delta_ts = map(γ->calculate_Δt_from_Δx(80/400,γ,D,r),all_gamma_indices)

p = plot_gamma_spread(variances,sums,all_gamma_indices[1:5],how_long,all_delta_ts;zoom_val=0.0,label_fit=true)
plot!(legend=:bottomright)

frac = 0.3
xb = 0.05
p = plot!(p, inset = (1,bbox(xb, xb, frac, frac, :top, :left)) )
p = draw_square_abs(p,[0,0],[1,1])
p = arrow_from_abs_to_frac(p,[0,1],[xb,1-(xb+frac)])
p = arrow_from_abs_to_frac(p,[1,1],[xb+frac,1-(xb+frac)])
p = plot!(p, mirror=true,subplot=2, frame=:box )

plot_gamma_spread(p,variances,sums,all_gamma_indices[1:5],how_long,all_delta_ts;
    zoom_val=100.0,label_fit=true,for_plot=(legend=:none,xlab="",ylab=""),subplot_index=2)

# +
folder_path = string(global_folder_path,"/",picture_folder_name)

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.5)


file_name = "AnomDiffSmallDx" # the one that takes long

save_fig(;dpi=dpi_, plot_size=size_, save_type="png",
                folder_path = folder_path, file_name = file_name,folder_name = "Singles", 
                for_plot=(legendfontsize=12,),scale_=1.0)
# -

# # Other plots 

# +
function log_(a,b,x)
    return a.*log.((b.*x).+1)
end

function power_(c,d,x)
    return c.*(x.^d)
end

function sqrt_(f,x)
    return power_(f,0.5,x)
end

# -

x = range(0,1,step=0.001)
plot(x,log_(5.54,11.5,x))
plot!(x,sqrt_(14.6,x))

# +
plot()
T = 50
X = 100
K = 10
dx = 0.5

eps = 0.0001
log_alpha_range = range(0.02,1,step=0.005)
alpha_range = range(0.5,1-0.00005,step=0.00005)

comput(x) = T * K * X / (dx)^(1+4/x)
scaled_comput(x) = x == 1 ? 1/700000 * comput(x)/K : 1/700000 * comput(x)

log_comput(x) = log(T) + log(K) + log(X) - (1+4/x) * log(dx) -  7
log_scaled_comput(x) = log_comput(x) 

plot(log_alpha_range, log_comput.(log_alpha_range),
    xlab=L"Anomalous diffusion scaling parameter $\alpha$",ylab=L"\ln(\textrm{Complexity}) \propto \ln(TKX/dx^{1+4/\alpha})",
    label="",color=1,
    yticks=range(0,150,step=20),xticks = range(0.0,1.0,step=0.1))
plot!(twinx(),alpha_range, scaled_comput.(alpha_range),
    ylab="Machine time [hours]",label="",color=2,
    yticks=range(0,40,step=5))
scatter!([0.8],[scaled_comput(0.8)],subplot=2,label="4.5 hours",color=2)
scatter!([1.0],[scaled_comput(1.0)],subplot=2,label="30 seconds",color=2,markershape=:square)
plot!(ylims = [0,150],subplot=1)

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "ComputComplex"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.4)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)

# +
plot()
T = 50
X = 100
K = 10
dx = 0.5

eps = 0.0001
log_alpha_range = range(0.2,1,step=0.0005)
alpha_range = range(0.8,1,step=0.00005)
scale = 10^4.15
convers = 700000 * scale

comput(x) = T * K * X / (dx)^(1+4/x) .* 3600 * scale
comput_k(x) = T * X / (dx)^(1+4/x) .* 3600 / 30 * scale
scaled_comput(x) = (comput(x) / convers)
scaled_comput_k(x) = (comput_k(x) / convers)


plot(log_alpha_range,(comput.(log_alpha_range)),
    xlab=L"Anomalous diffusion scaling parameter $\alpha$",ylab=L"Complexity ($\propto TKX/dx^{1+4/\alpha}$)",
    label="",color=1,yscale=:log10,yticks=myr10.((-1.0):20),grid=true)
plot!(twinx(),log_alpha_range, scaled_comput.(log_alpha_range),
    ylab="Machine time [seconds]",label="",color=2,yscale=:log10,yticks=myr10.((-1.0):20),grid=true)#,
  
scatter!([0.8],[comput(0.8)],subplot=1,label="4.5 hours",color=1)
scatter!([0.8],[scaled_comput(0.8)],subplot=2,label="4.5 hours",color=2)
scatter!([1.0],[comput_k(1.0)],subplot=1,label="30 seconds",color=1)
scatter!([1.0],[scaled_comput_k(1.0)],subplot=2,label="30 seconds",color=2)
plot!()

# +
folder_name = string(picture_folder_name,"/","Singles")
my_file_name = "ComputComplex"

(dpi_,size_) = get_size_and_dpi(;target_dpi=my_dpi,target_page_side=my_page_side,fraction_of_page=0.4)

save_fig(;folder_path=global_folder_path,folder_name=folder_name,file_name=my_file_name,
    save_type="png",dpi=dpi_,plot_size=size_,notify=true)
# -

#

# +
# Configuration Arguments
num_paths = 1#30

L = 200     # real system width (e.g. 200 meters)
M = 400     # divided into M pieces , 400

p₀ = 230.0  #this is the mid_price at t=0  238.75 

# Free-Parameters for gaussian version
D = 0.5#0.5/8 # real diffusion constant e.g. D=1 (meters^2 / second), 1
α = 0.0 # legacy, no longer used

ν = 2.0 #removal rate
γ = 1.0 #fraction of derivative (1 is normal diffusion, less than 1 is D^{1-γ} derivative on the RHS)

T = 3000

# Source term:
λ = 1.0 #
μ = 0.1 #
mySourceTerm = SourceTerm(λ, μ, true);

# Coupling term:
a = 13.0  #gap between stocks before at full strength: strong is 0.3
b = 1.0   #weighting of interaction term: strong is 2
c = 1.2   #skew factor: strong is 2

myCouplingTerm = CouplingTerm(μ, a, b, c, false);

# My randomness term
σ = 1.0 #variance in randomness
r = 0.5 #proportion of time in which it jumps left or right
β = 0.0 #probability of being the value of the previous lag or mean reversion strength
lag = 10 #lag
do_random_walk = false #behave like a random walk
myRandomnessTerm = RandomnessTerm(σ,r,β,lag,do_random_walk,false)

Δx = L / M  # real gap between simulation points 
Δt = (r * (Δx^2) / (2.0 * D))^(1/γ)

# RL Stuff:
RealKickStartTime = 8 # when, in real time, to kick the system
SimKickStartTime = to_simulation_time(RealKickStartTime,Δt)-2 # convert to simulation time
Position = 0
Volume = 20

myRLPusher = RLPushTerm(SimKickStartTime,SimKickStartTime+8,Position,Volume,true)

lob_model = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ, 
    mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm,shift=-1,old_way=true);

print((Δt,to_simulation_time(T,Δt),num_paths*to_simulation_time(T,Δt))) #about 2GB RAM per 100K, i.e. can only do about 1.8 million
lob_model.SK_DP
# -

# check something actually happens for one example
if true
    try clear_double_dict(Dat) catch e print("Not initialized") end
    GC.gc()
    
    lob_model = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ, 
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm,michaels_way=true);
    TrulyOldWay = InteractOrderBooks([lob_model], -1, true) ;
    
    lob_model = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ, 
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm,michaels_way=false,shift=-1,old_way=true);
    DatSpecialCase = InteractOrderBooks([lob_model], -1, true) ;
    
    lob_model = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ, 
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm,michaels_way=false,shift=0,old_way=false);
    DatNewWayWithNoShift = InteractOrderBooks([lob_model], -1, true) ;
    
    lob_model = SLOB(num_paths, T, p₀, M, L, D, ν, α, γ, 
        mySourceTerm, myCouplingTerm, myRLPusher, myRandomnessTerm,michaels_way=false,shift=-1,old_way=false);
    DatNewWayWithShift = InteractOrderBooks([lob_model], -1, true) ;

    
    sub_x = Int(3M/8):Int(5*M/8)
    how_many = 1
    p_arr1 = Array{Plots.Plot{Plots.GRBackend},1}(undef,how_many)
    start_pos =  SimKickStartTime+14
    #start_pos = to_simulation_time(T,Δt) - how_many - 1
    for i in [start_pos:(start_pos+how_many-1);]
        #p_arr1[i] = plot_density_visual(SimKickStartTime-2+i, to_real_time(SimKickStartTime-2+i,Δt), 1, Dat,false, true, 10, Dat[1][1].raw_price_paths[SimKickStartTime])
        
        plot()
        plot(TrulyOldWay[1][1].slob.x[sub_x], TrulyOldWay[1][1].lob_densities[sub_x,i], color=1,ls=:solid,lw=3,label="Old way (Michael's code exactly)");
        plot!(DatSpecialCase[1][1].slob.x[sub_x], DatSpecialCase[1][1].lob_densities[sub_x,i], color=2,ls=:dash,lw=2,label="New way analytically reduced to old (bound correction and only keep first order in Dt)")
        plot!(DatNewWayWithNoShift[1][1].slob.x[sub_x], DatNewWayWithNoShift[1][1].lob_densities[sub_x,i], color=3,ls=:dash,lw=2,label="New way with no bound correction (no bound correction and all orders in Dt)")
        plot!(DatNewWayWithShift[1][1].slob.x[sub_x], DatNewWayWithShift[1][1].lob_densities[sub_x,i], color=4,ls=:dash,lw=2,label="New way with bound correction (bound correction and all orders in Dt)")
        
        scatter!(TrulyOldWay[1][1].slob.x[sub_x], TrulyOldWay[1][1].lob_densities[sub_x,i], color=1,label="",markersize=4,markerstrokewidth=0.0);
        scatter!(DatSpecialCase[1][1].slob.x[sub_x], DatSpecialCase[1][1].lob_densities[sub_x,i], color=2,label="",markersize=4,markerstrokewidth=0.0)
        scatter!(DatNewWayWithNoShift[1][1].slob.x[sub_x], DatNewWayWithNoShift[1][1].lob_densities[sub_x,i], color=3,label="",markersize=4,markerstrokewidth=0.0)
        scatter!(DatNewWayWithShift[1][1].slob.x[sub_x], DatNewWayWithShift[1][1].lob_densities[sub_x,i], color=4,label="",markersize=4,markerstrokewidth=0.0)
        
        p_arr1[i-start_pos+1] = plot!()
    end
    plot(p_arr1...,size=(1200,1000))
    
    #png("/home/derickdiana/Desktop/Masters/Reworked/CompareDifferentSchemesWithoutRandomness.png")
    plot!()
end


# +
numeric_method_1 = function(x)
    return x*x
end

numeric_method_2 = function(x)
    return x*x*x
end

do_single_step = function(x)
    return chosen_numeric_method(x)
end

run_scheme = function(x)
    return do_single_step(x)
end

initialize_and_run = function(x,which)
    if which
        chosen_numeric_method = numeric_method_1
    else
        chosen_numeric_method = numeric_method_2
    end
    
    ##
    
    return run_scheme(x) 
end

# +
numeric_method_1 = function(x)
    return x*x
end

numeric_method_2 = function(x)
    return x*x*x
end

do_single_step = function(x,chosen_numeric_method)
    return chosen_numeric_method(x)
end

run_scheme = function(x,chosen_numeric_method)
    return do_single_step(x,chosen_numeric_method)
end

initialize_and_run = function(x,which)
    chosen_numeric_method = () -> print("should not have run me")
    if which
        chosen_numeric_method = numeric_method_1
    else
        chosen_numeric_method = numeric_method_2
    end
    
    ##
    
    return run_scheme(x,chosen_numeric_method) 
end

# +
numeric_method_1 = function(x)
    return x*x
end

numeric_method_2 = function(x)
    return x*x*x
end

do_single_step = function(x,which)
    if which
        return numeric_method_1(x)
    else
        return numeric_method_2(x)
    end
end

run_scheme = function(x,which)
    return do_single_step(x,which)
end

initialize_and_run = function(x,which)
    return run_scheme(x,which) 
end
# -

initialize_and_run(5,false)
