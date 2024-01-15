# -*- coding: utf-8 -*-
# # Other

# +
## Functions related to summing
function get_sums(lob_densities;absolute=true,raw_prices=[],Δx=0,base_x=0,do_half=false)
    l = size(lob_densities)[2]
    sums = zeros(Float64, l)
    
    if absolute 
        my_sum = (x) -> sum(abs.(x)) else
        my_sum = (x) -> sum(x)
    end
    
    for t in 1:l
        if do_half
            sub = 1:price_to_index(raw_prices[t],Δx,base_x)
        end
        
        sums[t] = my_sum(lob_densities[:,t])
    end
    
    return sums
end


function plot_sums(Dat;path_num=1,slob_num=1,new_plot=true,absolute=true)
    if new_plot
        plot()
    end
    sums = get_sums(Dat[path_num][slob_num].lob_densities;absolute=absolute)
    slob = Dat[path_num][slob_num].slob

    scatter!(sums,markersize=1.4,label="Area",size=(1400,500))
    vline!([slob.rl_push_term.StartTime],label="Kick time")
    if absolute
        hline!([sums[slob.rl_push_term.StartTime-1]+slob.Δt*slob.rl_push_term.Amount],label="Theoretical kick volume")
    else
        hline!([-slob.Δt*slob.rl_push_term.Amount],label="Theoretical kick volume")
    end
    
    source_sums = get_sums(Dat[path_num][slob_num].sources;absolute=absolute)
    hline!([source_sums[2]/slob.nu],label="Theoretical equilibrium volume") #2 because the source isn't there at time 1
    
    plot!(xlabel="Simulation steps",ylabel="Signed area under system")
    plot!(size=(1200,500))
end
# -

function my_repeat(func,n)
    res = Array{typeof(func())}(undef,n)
    for i in 1:n
        res[i] = func()
    end
    return res
end

# +
function arrow_from_abs_to_frac(plt,from_pos_abs,to_pos_frac;subplot=1,color=:black,include_dot=false)
    
    xmin = xlims(plt[subplot])[1]
    xmax = xlims(plt[subplot])[2]
    
    ymin = ylims(plt[subplot])[1]
    ymax = ylims(plt[subplot])[2]
    
    to_pos_abs = (xmin + to_pos_frac[1] * (xmax-xmin),ymin + to_pos_frac[2] * (ymax-ymin))
    plot!(plt,[from_pos_abs[1],to_pos_abs[1]],[from_pos_abs[2],to_pos_abs[2]],subplot=subplot,color=color,label="")
    
    if include_dot
        scatter!(plt,[from_pos_abs[1]],[from_pos_abs[2]],subplot=subplot,color=color,
            ms=3.0,markerstrokewidth=0.1,  ma=1)
    end
    
    return plt
    
end
# -

function draw_square_abs(plt,one_side,other_side)
    plt = plot!(plt,[one_side[1]  ,other_side[1]],[one_side[2]  ,one_side[2]  ],color="black",label="")
    plt = plot!(plt,[one_side[1]  ,other_side[1]],[other_side[2],other_side[2]],color="black",label="")
    plt = plot!(plt,[one_side[1]  ,one_side[1]  ],[one_side[2],other_side[2]],color="black",label="")
    plt = plot!(plt,[other_side[1],other_side[1]],[one_side[2],other_side[2]],color="black",label="")
    
    return plt
end

# +
## Functions to do math things
function get_second_derivative(x,temp)
    middle = 2:length(temp)-1
    return ((temp[middle.+1]-temp[middle])./(x[middle.+1]-x[middle])-(temp[middle]-temp[middle.-1])./(x[middle]-x[middle.-1]))./(x[middle.+1]-x[middle.-1])
end;

function myvar(p,x,Δx;option=1)
    if option == 1
        mu = mymean(p,x,Δx)
        Ex2 = 0
        for i in 1:length(p)
            Ex2 += p[i]*x[i]^2
        end
        return Ex2 - mu^2
    else
        n = Distributions.fit_mle(Normal,x,p)
        return (n.σ)^2
    end
end

function mymean(p,x,Δx;option=1)
    if option == 1
        sum = 0
        for i in 1:length(p)
            sum += p[i]*x[i]
        end
        return sum
    else
        n = Distributions.fit_mle(Normal,x,p)
        return n.μ
    end
end
# -

function my_power_fit(x,y)
    m(t, p) = p[1] .* (t .^ p[2])
    p0 = [1.0, 0.7]
    fit = LsqFit.curve_fit(m, x, y, p0)
    
    a,b = fit.param
    covar = estimate_covar(fit)
    au,bu = sqrt.([covar[1,1],covar[2,2]])
    return ((a,b),(au,bu))
end

function my_log_fit(x,y)
    m(t, p) = p[1] .+ (p[2].*log.(t))
    p0 = [0.0, 2.0]
    fit = LsqFit.curve_fit(m, x, y, p0)
    
    a,b = fit.param
    covar = estimate_covar(fit)
    au,bu = sqrt.([covar[1,1],covar[2,2]])
    return ((a,b),(au,bu))
end

function my_log_fit2(x,y)
    m(t, p) = p[2].*(log.((p[1].*t).+1))
    p0 = [0.2, 2.0]
    fit = LsqFit.curve_fit(m, x, y, p0)
    
    a,b = fit.param
    covar = estimate_covar(fit)
    au,bu = sqrt.([covar[1,1],covar[2,2]])
    return ((a,b),(au,bu))
end

function my_log_fit3(x,y)
    m(t, p) = (p[3]).+(p[2].*(log.((p[1].*t).+1)))
    p0 = [0.2, 2.0, 0.3]
    fit = LsqFit.curve_fit(m, x, y, p0)
    
    a,b,c = fit.param
    covar = estimate_covar(fit)
    au,bu = sqrt.([covar[1,1],covar[2,2]])
    return ((a,b),(au,bu))
end

function get_length_of_time_that_allows_kick(RealKickStartTime,l,Δt,seed)
    
    T = RealKickStartTime+(to_real_time(l,Δt)+1) #guess at total time needed

    Δts = generate_Δts_exp(T,Δt;seed=seed); #tgenerate random steps that reach at least that time
    
    SimKickStartTime = to_simulation_time(RealKickStartTime,Δt; do_exp = true, cum_Δts = cumsum(Δts)) # calculate when, in the random steps, your kick corresponds to
    
    SimLength = to_simulation_time(T,Δt; do_exp = true, cum_Δts = cumsum(Δts)) # calculate the length of the simulation in simulation steps

    mult = 1
    while (SimLength - SimKickStartTime) < (l+2) #if the simulation doesn't last long enough after the kick, make it longer
        mult = mult * 2

        T = RealKickStartTime+mult*(to_real_time(l,Δt)+1) #add extra time to the end

        Δts = generate_Δts_exp(T,Δt;seed=seed); #then generate random steps that reach at least that time
        
        SimKickStartTime = to_simulation_time(RealKickStartTime,Δt; do_exp = true, cum_Δts = cumsum(Δts)) # then calculate when, in the random steps, your kick corresponds to   
        
        SimLength = to_simulation_time(T,Δt; do_exp = true, cum_Δts = cumsum(Δts)) # calculate the length of the simulation in simulation steps
    end
    
    return (T,Δts,SimKickStartTime)
end

# +
function difference_checker(Dat1,Dat2)
    N = Dat1[1][1].slob.N
    
    mt = 1
    d1 = Dat1[1][1].lob_densities[:,mt]
    d2 = Dat2[1][1].lob_densities[:,mt]
    first_diff_pos = findfirst(x -> x .!= 0, d2.-d1)

    while (first_diff_pos == nothing) && mt < N
        mt += 1 

        d1 = Dat1[1][1].lob_densities[:,mt]
        d2 = Dat2[1][1].lob_densities[:,mt]
        first_diff_pos = findfirst(x -> x .!= 0, d2.-d1)
    end


    print("First difference occures at time ", mt ,"\n")
    print("First difference occures at position ", first_diff_pos ,"\n")
    print("Sums are ",sum(d1)," and ",sum(d2),"\n")
    print("Difference is ",d1[first_diff_pos] - d2[first_diff_pos],"\n")
    print("Next pos diff is ",d1[first_diff_pos+1] - d2[first_diff_pos+1],"\n")
    print("Prev pos diff is ",d1[first_diff_pos-1] - d2[first_diff_pos-1])
    
end

# +
function derivative_at(φ,i,Δx)
    return (φ[i+1]-φ[i])/(1.0*Δx)
end

function get_central_derivative(Dat)
    mydat = Dat[1][1]
    derivatives = repeat([0.0],mydat.slob.N)
    
    for t in 1:mydat.slob.N
        φ = mydat.lob_densities[:,t]
        p = mydat.raw_price_paths[t]
        i = price_to_index(p,mydat.slob.Δx,mydat.slob.x[1])+1
        
        derivatives[t] = derivative_at(φ,i,mydat.slob.Δx)
    end
    
    return derivatives
end

function get_effective_market_orders(Dat;shift=1)
    mydat = Dat[1][1]
    marketorders_l = repeat([0.0],mydat.slob.N)
    marketorders_r = repeat([0.0],mydat.slob.N)
    r = mydat.slob.randomness_term.r
    
    for t in 1:mydat.slob.N
        φ  = mydat.lob_densities[:,t]
        p  = mydat.raw_price_paths[t]
        
        P  = mydat.Ps[t]
        P⁺ = mydat.P⁺s[t]
        P⁻ = mydat.P⁻s[t]
        
        i  = price_to_index(p,mydat.slob.Δx,mydat.slob.x[1])+1
        
        # the below 2 lines assume the left hand side is the positive 1
        φ⁺ = φ[i]
        φ⁻ = φ[i+1]
        
        @assert φ⁺ * φ⁻ <= 0
        
        l_s = -r/2*φ⁻*shift #left shift
        r_s = -r/2*φ⁺*shift #left shift
        marketorders_l[t]  = max(P⁻*φ⁻+l_s, -P*φ⁺+l_s)
        marketorders_r[t]  = min(P⁺*φ⁺+r_s, -P*φ⁻+r_s)
    end
    
    return [marketorders_l.*mydat.slob.Δx,marketorders_r.*mydat.slob.Δx]
end

# +
function find_first_digit_power(a)
    n = 0
    if a == 1
        return n #i.e. zero
    elseif a < 1
        while a < 1
            a *= 10
            n += 1
        end
        return n
    else
        while a > 1
            a /= 10
            n += 1
        end
        return -(n-1)
        
    end
        
    
end
# -

function f_unc(a,au;n=2,b=true)
    pos = find_first_digit_power(au)+n-1
    a_ = mr(a,pos)
    au_ = mr(au,pos)
    if b
        (lb,rb) = ("(",")")
    else
        (lb,rb) = ("","")
    end
    return string(L"%$lb%$a_\pm%$au_%$rb")
end

function mr(a,d)
    return round(a,digits=d)
end

function my_pad(str::String, target_length::Int; do_left=true::Bool)
    l = length(str)
    
    if l < target_length
        diff = target_length - l
        add_space = repeat(" ",diff)
        
        if do_left
            return add_space*str
        else
            return str*add_space
        end
    end
    
    return str
end

""" 
    obtain_seed(seed::Int64=-1)

If seed is -1, then get a random seed. Otherwise, seed has already been chosen
so just return it. E.g. `seed = obtain_seed(seed)`
"""
function obtain_seed(seed=-1) 
    if seed == -1
        seed = floor(Int64,rand() * 1e8)
    end
    
    return seed
end

"""
    get_layout_for_n_plots(n::Int64,size_of_each::Tuple(Int64,Int64))

Calculate the size that a single plot will need to be to show `n` inner plots if each inner plot has
a size (2 element tuple) of `size_of_each`
"""
function get_layout_for_n_plots(n,size_of_each)
    cols = ceil(Int64, sqrt(n))
    rows =  floor(Int64,sqrt(n))

    dim = (cols * size_of_each[1],rows * size_of_each[2]) 
    layout = (rows,cols)
    
    return (layout,dim)
end


# # Visualizers

# ## Price impact

# +
"""
    fit_and_plot_price_impact( volumes::Array{Float64}, mean_price_impacts::Array{Float64}, var_price_impacts::Array{Float64}, labels::Array{String}; <keyword arguments>)

Given the results of calling the function `calculate_price_impacts` which returns the `mean_price_impacts` and `var_price_impacts` for each of the `volumes` used, pass them to this function
 to have them plotted. When plotting them, give them custom labels to go in the legend by passing `labels`.

# Keyword arguments
- `sub::Array{Float64}=-1`: Which subset of the volumes to plot. Will be run as `volumes = volumes[sub]`. Allows for trying different subsets quickly.
- `do_kinks::Array{Float64}=true`: Whether to highlight kinks on the plot by using arrows.
- `colors::Array{Float64}=-1`: What colors to assign to each line in the order that they are drawn.
- `do_power_fit::Bool=false`: Whether to do a fit using a function of the form "y = ax^b" and include it on the plot.
- `xticks::Array{Float64}=-1`: Which 'xtick' positions to highlight. Will be passed to `plot` as `plot(;xticks=xticks)`.
- `new_plot::Bool=false=-1`: Whether to create a new plot or draw to an existing one.
- `for_plot::Tuple(...)=()`:
- `do_ribbon::Bool=false`:
- `do_horiz_log_shift::Bool=true`:
- `do_log_fit::Bool=true`:
- `do_log_plot::Bool=false`:
- `do_vert_log_shift::Bool=true`:
- `use_only_straight_region::Bool=false`:
- `shift_y_ticks::Bool=false`:
- `do_just::Int64=-1`:
- `straight_line_frac_to_plot::Float64=-1`:
- `square_the_price::Bool=false`:
"""

# here we must have that if (li in my_labels) then steps[li] is the step to be used when drawing line li
function fit_and_plot_price_impact(plt,(steps,volumes,labels),mean_price_impacts,var_price_impacts;
                                                sub=-1,do_kinks=true,colors=-1,do_power_fit=false,xticks=-1,forplot=(),
                                                do_ribbon=true,do_horiz_log_shift=true,do_log_fit=true,do_log_plot=false,
                                                do_vert_log_shift=true,use_only_straight_region=false,shift_y_ticks=false,do_just=-1,
                                                straight_line_frac_to_plot=-1,square_the_price=false,fit_labels_in_legend=true,
                                                modify_input_v=identity,modify_input_p=identity,
                                                modify_plot_v=identity, modify_plot_p=identity,type_1_log=true,subplot_index=-1,tiny_shift=0.0)
    # the below is just for readability
    
    modify_plot__v = modify_plot_v
    modify_plot__p = modify_plot_p
    
    mr(v,a) = round(v,digits=a)
    
    subplt = ()
    if subplot_index>0
        subplt = (subplot=subplot_index,)
    end
    
    # Initialize stuff
    #########################################################################
    li_len = length(labels)
    vi_len = length(volumes)
    
    if length(steps)==1
        steps = repeat(vcat(steps),li_len)
    end
    
    
    if typeof(do_power_fit)==Bool
        do_power_fit = repeat([do_power_fit],li_len)
    end
    
    if typeof(do_log_fit)==Bool
        do_log_fit = repeat([do_log_fit],li_len)
        print(do_power_fit)
    end
    
    if do_just==-1
        do_just = repeat([true],li_len)
    end
    
    if sub==-1
        sub = 1:vi_len
    end
    
    if straight_line_frac_to_plot == -1
        straight_line_frac_to_plot = repeat([1.0],li_len)
    end
    
    if colors==-1 #change later!
        if li_len<=5
            colors = ["blue","red","green","orange","purple"]
        else
            colors = 1:li_len
        end
    end
    
    shift = do_horiz_log_shift ? 1 : 0
    
    if do_log_plot  && (modify_plot__v == identity)
    #    modify_plot__v = (v) -> log(v+1)
    end
    #########################################################################
    
    volumes_input = modify_input_v.(volumes[sub]) #the original volumes given before manipulation
    volumes_plot  = modify_plot__v.(volumes[sub])#log.(volumes.+1)[:]
    
    for ind in 1:li_len
        l = steps[ind]
        
        if !do_just[ind] 
            continue
        end
        
        #mean_price_imp = mean_price_impacts#[l,:,:]
        #var_price_imp  =  var_price_impacts#[l,:,:]
        
        mean_price_input    =  modify_input_p.(mean_price_impacts[l,sub,ind]) 
        mean_price_plot     =  modify_plot__p.(mean_price_impacts[l,sub,ind]) 

        var_price_input     =  modify_input_p.( var_price_impacts[l,sub,ind]) 
        var_price_plot      =  modify_plot__p.( var_price_impacts[l,sub,ind])
        
        # the below sets the upper cutoff to either the highest volume or the point with the largest derivative
        upper_fit_cut_off = length(sub) #upper cut off is highest volume
        if use_only_straight_region #upper cutoff is point with largest derivative
            upper_fit_cut_off = findmax(diff(mean_price_input))[2]
            upper_fit_cut_off = upper_fit_cut_off == nothing ? 0 : upper_fit_cut_off
        end
        sub_to_fit = sub[1:upper_fit_cut_off]
        
        
        # the below fits a log function to the input data
        if do_log_fit[ind]
            if type_1_log
                (a,b),(au,bu) = my_log_fit( volumes_input[sub_to_fit].+shift ,  mean_price_input[sub_to_fit] )
            else
                (a,b),(au,bu) = my_log_fit2( volumes_input[sub_to_fit] ,  mean_price_input[sub_to_fit] )
            end
            a = do_vert_log_shift ? a : 0

            
            line_plot_cut_off = straight_line_frac_to_plot[ind] * upper_fit_cut_off
            line_plot_cut_off = min( floor(Int64,line_plot_cut_off) ,  length(sub))
            sub_to_plot = sub[1:line_plot_cut_off]
            
            if type_1_log
                y_line_log_fit = a.+b.*log.(volumes_input[sub_to_plot].+shift)
            else
                y_line_log_fit = b.*log.((a.*volumes_input[sub_to_plot]).+shift)
            end
            
            curr_label = "Log fit"
            if fit_labels_in_legend
                f_unc_sa = f_unc(a,au)
                f_unc_sb = f_unc(b,bu)
                #curr_label = L"Log fit: %$f_unc_sa $+$ %$f_unc_sb $\log(V/A+1)$"
                curr_label = L"Log fit: %$f_unc_sb $\log$( %$f_unc_sa $V+1$)"
            end
            
            plt = plot!(plt,     volumes_plot[sub_to_plot] , modify_plot__p.(y_line_log_fit) ,
                           label=curr_label,w=1.5,color=colors[ind];subplt...)
            
        end
        
        # the below fits a power function to the input data
        if do_power_fit[ind]
            #c,d = power_fit(volumes_input[sub_to_fit],mean_price_input[sub_to_fit])
            (c,d),(cu,du) = my_power_fit(volumes_input[sub_to_fit],mean_price_input[sub_to_fit])
            
            y_line_power_fit = c.*(volumes_input[sub_to_fit]).^d
            
            curr_label = "Power fit"
            if fit_labels_in_legend
                f_unc_sc = f_unc(c,cu)
                f_unc_sd = f_unc(d,du)
                curr_label = L"Power fit: %$f_unc_sc $(V/A)$^%$f_unc_sd"
            end
                
            plt = plot!(plt, volumes_plot[sub_to_fit], modify_plot__p.(y_line_power_fit),
                label=curr_label,
                        w=1.5,color=colors[ind],linestyle=:dash;subplt...)
        end
        
        # plots the actual price impact points
        plt = scatter!( plt,     volumes_plot,   mean_price_plot,
                label=labels[ind],ms=1.2,markerstrokewidth=0.01,  ma=1,color=colors[ind];subplt...)
        
        # plots a ribbon of uncertainty around the price impact points
        if do_ribbon
            plt = plot!(plt,     volumes_plot,   mean_price_plot,  ribbon=var_price_plot.^0.5,alpha=0,
                fillalpha=0.4,fillcolor=colors[ind],label="";subplt...)
        end
            
        # if xticks not provided, use global xticks??????
        if xticks!=-1
            plot!(xticks=xticks;subplt...)
        end


        # detect the kinks in the graph
        if do_kinks
            
            vol_scale = (volumes_input[end]-volumes_input[1])/20
            impact_scale = (maximum(mean_price_input,dims=1)[1]-minimum(mean_price_input,dims=1)[1])/30
            
            second_deriv = get_second_derivative(volumes_input,mean_price_input)
            kink_position = findnext(x->x>0,second_deriv,1)
            
            kink_counter = 1
            
            while !(kink_position===nothing) && kink_counter<=20
                target_x, target_y = (volumes_input[kink_position+1],mean_price_impacts[kink_position+1])
                
                quiver!(plt,[target_x+vol_scale],[target_y-impact_scale],quiver=([-vol_scale],[impact_scale]),color=colors[ind];subplt...)
                scatter!(plt,[target_x],[target_y],markershape=:star5,color=colors[ind],
                    label=string("Kink at position ",round(target_x,digits=2)),markerstrokewidth=0.1,  ma=1;subplt...)
                
                kink_position = findnext(x->x>0,second_deriv,kink_position+2)
                
                kink_counter += 1
            end
        end

    end
    
    # label the xaxis
    my_xlabel = do_log_plot ? "log(Volume)" : "Volume"
    plot!(plt, xlabel=my_xlabel,ylabel=L"Price impact i.e. $p(t+1)-p(t)$";forplot...,subplt...)
end

#fit_and_plot_price_impact((steps,volumes,labels),mean_price_impacts,var_price_impacts) = 
#fit_and_plot_price_impact(plot(),(steps,volumes,labels),mean_price_impacts,var_price_impacts) 
# -


# ## Density

# +
"""
    plot_density_visual(Simulation_Time, Which_SLOB, Data_object; <keyword args>)

A core function for visualizing the order book. Returns a plot (by default centered at the intercept of the order book with the x-axis) 
which shows 7 lines:

-(1) The LOB (or LatOB) density itself\n
-(2) Its left approximated version for the exponential algorithm\n 
-(3) Its right approximated version for the exponential algorithm\n 
-(4) Couplings\n
-(5) Arrivals\n
-(6) Input from RL source\n
-(7) Removals\n

In addition it shows the intercept (i.e. price) and Δxₘ shifts away from the price. It does this at Simulation\\_Time. As the Data\\_object 
has many different SLOBs inside of it, one needs to specify Which\\_SLOB to use.

# Arguments
- `dosum::Bool=false`: When providing a Data object which contains two related SLOBs, visualize their sum rather than any individual one
- `plot_raw_price::Bool=true`: Place a dot at the intercept of the LOB with the x-axis
- `x_axis_width::Float64=-1`: Distance from the intercept to the edges of the plot (how much of the x-axis is seen). If left as -1, it defaults to 1/5 of the total simulated x-axis
- `center::Int64=-2`: When set to -2, the view is centered at the intercept at the beginning of the simulation
                      When set to -1, the view is centered at the current intercept.
                      When set to any positive value, the view is centered at that positive value.
- `shift_with_loop:Bool=false`: Only matters if center==-1. If true, then when the current intercept loops round due to cyclic boundary conditions, the view also loops around.
- `marker_size::Float64=1.6`: The marker size used for the intercept and Δxₘ displacement.
- `overall_alpha::Array{Float64}`: An array of 7 float values in the range [0,1] which determines the alpha value of graph of each of the 7 displayed lines in order (see above).
- `do_interp::Bool=false`: Whether to do a smooth interpolation between the points using the same algorithm as used to determine the smooth price impact.
- `path_to_plot::Int64=1`: Data object may have many paths, one needs to specify which to plot.
- `size::Tuple(Int64,Int64)`: The size of the outputted plot.
- `kw_for_plot::Tuple(...)`: A tuple with keywords which will be passed at the very end of this function to the 'plot' function as `plot(;kw_for_args...)`. 
                             Thus one may do, e.g: `kw_for_plot=(xlabel="x",)` which results in `plot(;xlabel="x")` via the `...` operation.
- `do_left::Bool=true`: Whether to include (2) in the above list of 7 lines
- `do_right::Bool=true`: Whether to include (3) in the above list of 7 lines
- `label_price::Bool=true`: Whether to include a label on the plot which reads "p=..." and shows the value of the intercept.
- `label_Δxₘ::Bool=true`: Whether to include a label on the plot which reads "Δxₘ=..." and shows the value of Δxₘ at this time.
- `annotate_pos::PositionKeyword=:topright`: The position of the two above mentioned labels. Possible inputs are the same as those 
                                             used to define legend position in Plot e.g. `:topleft, :top, :left, :bottomright:, :bottom, :right`
"""

function plot_density_visual(Dat, s, lob_num; 
                dosum=false, plot_raw_price=true, x_axis_width = -1, center = -2, shift_with_loop=false, marker_size=1.6,overall_alpha=[1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                do_interp=-1, path_to_plot=1, size=(1000,1000), kw_for_plot=(), shift_back_approx=true, do_left=true,do_right=true,label_price=true,label_Δxₘ=true,
                annotate_pos=:topright)
    
    
    dat = Dat[path_to_plot][lob_num]
    lob_model = dat.slob
    
    if !lob_model.store_past_densities
        @warn "Should not be using this function since you did not store the densities\n"
    end
    
    if do_interp==-1
        do_interp = lob_model.do_interp
    end
   
    mult = lob_model.Δt
    
    shift = dat.x_shifts[s]
    
    if center == -1
        center = dat.raw_price_paths[s]
    end
    
    if center == -2
        center = lob_model.p₀
    end
    
    if shift_with_loop
        camera_shift = shift
    else
        camera_shift = 0
    end
    
    if x_axis_width==-1
        x_axis_width = dat.slob.L/15
    end
    
    if lob_model.old_way
        removal_fraction = - lob_model.nu * lob_model.Δt
    else
        removal_fraction = exp(-lob_model.nu * lob_model.Δt) - 1
    end
    
    
    x_axis  = [center-x_axis_width + camera_shift,center+x_axis_width + camera_shift]
    
    lob_densities   = dat.lob_densities[:,s]
    lob_densities_L = dat.lob_densities_L[:,s]
    lob_densities_R = dat.lob_densities_R[:,s]
    couplings       = dat.couplings[:,s+1]#-1 not there before
    sources         = dat.sources[:,s+1]#-1 not there before
    rl_pushes       = dat.rl_pushes[:,s+1]#-1 not there before
    removals        = dat.lob_densities[:,s]
    
    if dosum
        dat2 = Dat[path_to_plot][3-lob_num]
        lob_densities   .+= dat2.lob_densities[:,s]
        lob_densities_L .+= dat2.lob_densities_L[:,s]
        lob_densities_R .+= dat2.lob_densities_R[:,s]
        couplings       .+= dat2.couplings[:,s+1]#-1 not there before
        sources         .+= dat2.sources[:,s+1]#-1 not there before
        rl_pushes       .+= dat2.rl_pushes[:,s+1]#-1 not there before
        removals        .+= dat2.lob_densities[:,s]
    end
    
    lob_densities = lob_densities
    lob_densities_L = lob_densities_L
    lob_densities_R = lob_densities_R
    couplings = mult.*couplings
    sources = mult.*sources
    rl_pushes = mult.*rl_pushes
    removals = removal_fraction.*removals
    
    x_range = lob_model.x .+ shift 
    common = (label="",markersize=marker_size,markerstrokewidth=0.0)
    
    
    Δx_ = dat.slob.Δxs[s-1]
    
    if shift_back_approx
        x_range_shifted_left = x_range.-Δx_
        x_range_shifted_right = x_range.+Δx_
    else
        x_range_shifted_left = x_range
        x_range_shifted_right = x_range
    end
    
    plt = 
     scatter(x_range,                lob_densities,   color=1, alpha = overall_alpha[1];common...); 
    if do_left
    scatter!(x_range_shifted_left,   lob_densities_L, color=1, alpha = overall_alpha[2];common...); end
    if do_right
    scatter!(x_range_shifted_right,  lob_densities_R, color=1, alpha = overall_alpha[3];common...); end
    scatter!(x_range,                couplings,       color=2, alpha = overall_alpha[4];common...) ;
    scatter!(x_range,                sources,         color=3, alpha = overall_alpha[5];common...) ;
    scatter!(x_range,                removals,        color=5, alpha = overall_alpha[6];common...) ;
    scatter!(x_range,                rl_pushes,       color=4, alpha = overall_alpha[7];common...) ;
    
    check_zeros = (s==1)
    if do_interp
        x_range = lob_model.x_range#lob_model.x[1]:(lob_model.L/lob_model.M):lob_model.x[end]
        x_range_dense = lob_model.x[1]:(lob_model.L/lob_model.M/10):lob_model.x[end] .+ shift 
        
        lob_densities   = auto_interpolator(x_range,lob_densities,  x_range_dense;check_zeros=check_zeros)
        if do_left
        lob_densities_L = auto_interpolator(x_range,lob_densities_L,x_range_dense;check_zeros=check_zeros) end
        if do_right
        lob_densities_R = auto_interpolator(x_range,lob_densities_R,x_range_dense;check_zeros=check_zeros) end
        sources         = auto_interpolator(x_range,sources,        x_range_dense;check_zeros=check_zeros)
        couplings       = auto_interpolator(x_range,couplings,      x_range_dense;check_zeros=check_zeros)
        removals        = auto_interpolator(x_range,removals,       x_range_dense;check_zeros=check_zeros)
        rl_pushes       = auto_interpolator(x_range,rl_pushes,      x_range_dense;check_zeros=check_zeros)
    else
        x_range_dense = lob_model.x .+ shift
    end
    
    if shift_back_approx
        x_range_shifted_left_dense = x_range_dense.-Δx_
        x_range_shifted_right_dense = x_range_dense.+Δx_
    else
        x_range_shifted_left_dense = x_range_dense
        x_range_shifted_right_dense = x_range_dense
    end
    
    
    plot!(x_range_dense,               lob_densities,   label="φⁱ",      color=1,alpha = overall_alpha[1]); 
    if do_left
    plot!(x_range_shifted_left_dense,  lob_densities_L, label="φⁱ⁻¹",    color=1,alpha = overall_alpha[2],style=:dash); end
    if do_right
    plot!(x_range_shifted_right_dense, lob_densities_R, label="φⁱ⁺¹",    color=1,alpha = overall_alpha[3],style=:dashdotdot); end
    #plot!(x_range_dense,           couplings, color=2, label="Coupling",alpha = overall_alpha[4]) ;
    plot!(x_range_dense,               sources,         label="Arrivals", color=3,alpha = overall_alpha[5]) ;
    plot!(x_range_dense,               removals,        label="Removals", color=5,alpha = overall_alpha[6]) ;
    plot!(x_range_dense,               rl_pushes,       label="Impulse",  color=4,alpha = overall_alpha[7]) ;
    
    
    price_pos = dat.raw_price_paths[s]
    if (plot_raw_price)
        if (isinteger(s*lob_model.Δt))
            mycol = :black
        else 
            mycol = 1
        end
        
        
        scatter!([price_pos]    ,[0],label="p";                   
                                                    markersize=3,markercolor= mycol ,markerstrokewidth=0.5)
        if do_left
        scatter!([price_pos-Δx_],[0],label="p-Δxₘ"     ;markershape=:star4,
                                                    markersize=5,markercolor="black",markerstrokewidth=0.5) end
        if do_right
        scatter!([price_pos+Δx_],[0],label="p+Δxₘ"     ;markershape=:star4,
                                                    markersize=5,markercolor="black",markerstrokewidth=0.5) end
    end
    
    if lob_model.do_exp_dist_times
        real_time = round(lob_model.Δts_cum[s],digits=2)#round(lob_model.Δt * (s-1),digits=3)
    else
        real_time = round(lob_model.Δt * (s-1),digits=3)
    end
    
    top_right_label = ""
    if label_price
        top_right_label *= my_pad("  p   = ",8;do_left=false) * my_pad(string(round(price_pos,digits=2)),8;do_left=false)
    end
    top_right_label     *= "\n"
    if label_Δxₘ
        top_right_label *= my_pad("Δxₘ = ",8;do_left=false) * my_pad(string(round(Δx_,digits=2)),8;do_left=false)
    end
    top_right_label     *= "\n"
    
    annotate!((annotate_pos,text(top_right_label,8)))
    
    plot!( legend=:bottomleft, title="", xlab="Price at time=$real_time", ylab="Densities",xlim=x_axis)#, legendfontsize=17.0)
    plot!(;kw_for_plot...)
    return plt
end;
# -

# ## Price change

# +
# here we must have that if (li in my_labels) then volumes[li] is the volume to be used when drawing line li
function fit_and_plot_price_change((steps,volumes,my_labels),mean_price_impacts,var_price_impacts;
                                        new_plot=true,colors=-1,do_just=-1,forplot=(),kick_end_time=-1,
                                        modify_input_t=identity,modify_input_p=identity,
                                        modify_plot_t=identity, modify_plot_p=identity)
    
    # the below is just for readability
    modify_plot__t = modify_plot_t
    modify_plot__p = modify_plot_p
    
    
    li_len = length(my_labels)
    ti_len = size(mean_price_impacts)[1]
    
    if length(volumes)==1
        volumes = repeat(vcat(volumes),li_len)
    end
    
    if new_plot #want a new plot?
        plot()
    end
    
    if do_just==-1
        do_just = repeat([true],li_len)
    end
    
    if colors==-1 #change later!
        if li_len<=5
            colors = ["blue","red","green","orange","purple"]
        else
            colors = 1:li_len
        end
    end
    
    sub = 1:ti_len
    
    steps_input = modify_input_t.(steps) #the original steps given before manipulation
    steps_plot  = modify_plot__t.(steps)#log.(volumes.+1)[:]
    
    for ind in 1:li_len
        v = volumes[ind]
        
        if !do_just[ind]
            continue
        end
        
        mean_price_input    =  modify_input_p.(mean_price_impacts[sub,v,ind])
        mean_price_plot     =  modify_plot__p.(mean_price_impacts[sub,v,ind])

        var_price_input     =  modify_input_p.( var_price_impacts[sub,v,ind])
        var_price_plot      =  modify_plot__p.( var_price_impacts[sub,v,ind])
        
        
        scatter!(steps_plot,mean_price_plot,label=my_labels[ind],
        ms=1.5,markerstrokewidth=0.1,  ma=1, color=colors[ind])
        plot!(steps_plot,mean_price_plot,label="",color=colors[ind])
    end
    
    if kick_end_time!=-1
        vline!([kick_end_time],label="End of meta-order")
    end
    
    plot!(xlabel="Time (t)",ylabel="Price (p(t))";forplot...)
    
end
# -

# ## Path

function show_path(Dat;num_steps=-1, fps_target=-1, time_should_run_for = -1,
                    start_time = -1, end_time = -1, 
                    start_time_sim = -1, end_time_sim = -1, path_to_plot=1, kw_for_visual=(), kw_for_path=()) 
    
    slob = Dat[path_to_plot][1].slob
    
    
    if end_time_sim == -1       #if they didn't already give an end time in simulation time, calculate one
        if end_time == -1       #if they didn't set any end time in real time, assume it is the real end of time (T)
            end_time = slob.T
        end                     #after this, end_time is always set
        end_time_sim = to_simulation_time(end_time,slob.Δt;do_exp=slob.do_exp_dist_times,cum_Δts=slob.Δts_cum)
    end
    
    if start_time_sim == -1       #if they didn't already give an start time in simulation time, calculate one
        if start_time == -1       #if they didn't set any start time in real time, assume it is the real start of time (0)
            start_time = 0
        end                       #after this, start_time is always set
        start_time_sim = to_simulation_time(start_time,slob.Δt;do_exp=slob.do_exp_dist_times,cum_Δts=slob.Δts_cum)
        if start_time_sim == 1 
            start_time_sim += 1
        end
    end
    
    total_length_sim = end_time_sim - start_time_sim
    
    if start_time_sim == 0 #real time starts at 0... 
        start_time_sim = 1 #...the array starts at 1
    end
    
    if num_steps == -1                        #if they didn't set any number of steps, pick 100
        num_steps = min(total_length_sim,100) #but don't pick 100 if the entire simulation has less than 100 steps total
    else
        num_steps = min(total_length_sim,num_steps) #ensure they didn't ask for more steps than we have
    end
    
    step = floor(Int,total_length_sim/num_steps)

    myrange = (start_time_sim):step:((start_time_sim+total_length_sim)-4*step)
    
    p_outer = Progress(length(myrange),dt=0.1)
    
    anim = @animate for s = myrange           
        
        plts = Array{Plots.Plot{Plots.GRBackend},1}(undef,6)
        
        r = to_real_time(s, slob.Δt;do_exp=slob.do_exp_dist_times,cum_Δts=slob.Δts_cum)          #r is the time in real time
        
        plts[1] = plot_price_path(plot(),Dat,s,1,false;path_to_plot=path_to_plot,kw_for_path...)

        if length(Dat[path_to_plot])>1
            plts[3] = plot_price_path(plot(),Dat,s,2,false;path_to_plot=path_to_plot,kw_for_path...)
            plts[5] = plot_price_path(plot(),Dat,s,1,true;path_to_plot=path_to_plot,kw_for_path...)
        end

        if slob.store_past_densities
            
            plts[2] = plot_density_visual(Dat,s,1;path_to_plot=path_to_plot,kw_for_visual...)
            
            if length(Dat[path_to_plot])>1
                plts[4] = plot_density_visual(Dat,s,2;path_to_plot=path_to_plot,kw_for_visual...)
                plts[6] = plot_density_visual(Dat,s,1; dosum=true, plot_raw_price=false,path_to_plot=path_to_plot,kw_for_visual...)
            end
            
        end
        
       
        plts = [plts[i] for i in 1:length(plts) if isassigned(plts, i)] 
        
        ################################
        densities = 1 + 1*((slob.store_past_densities) ? 1 : 0)
        multiple =  1 + 2*((length(Dat[path_to_plot])>1)          ? 1 : 0)
        
        dim = (densities * 500,multiple * 400) 
        l = (multiple,densities)
        ################################
        
        plot(plts...,layout = l,size=dim)

        next!(p_outer)
    end
    
    # In the below, since (fps) = (num_steps)/(time_should_run_for), the user cannot pick all three. So we choose to prioritize the time it should run for
    
    if (time_should_run_for != -1) #if they specified a time the animation should run for, pick the fps appropriately
        fps_target = num_steps/time_should_run_for 
    else   
        if (fps_target == -1) #the only way this runs is if both the fps_target and time_it_should_run_for are not set. If they chose an fps_target, that is used
            time_should_run_for = 5 #choose 10 seconds as a default run time
            fps_target = num_steps/time_should_run_for 
        end
    end

    return gif(anim, "/tmp/LOB.gif", fps=fps_target)
    
    #gif(anim, "~/Desktop/Masters/StillWorking/Random_Walk_And_Coupling_For_Alpha_Is_0.7.gif", fps=20*length(myrange)/200)
end

# +
function plot_price_path(plt, Dat, s, lob_num, diff;
        path_to_plot = 1,kw_for_plot=(),do_scatter=false,subplot=1,do_extra_bit=false)
    lob_model = Dat[1][lob_num].slob
    
    subplt = ()
    if subplot > 1
        subplot_ = subplot
        subplt = (subplot = subplot_,)
    end
    
    r = to_real_time(s, lob_model.Δt;do_exp=lob_model.do_exp_dist_times,cum_Δts=lob_model.Δts_cum)+1          #r is the time in real time
    
    alpha = 0.2
    malpha = 0.1
    
    
    for path in 1:lob_model.num_paths
        raw_price_paths = Dat[path][lob_num].raw_price_paths[1:s]
        if diff
            raw_price_paths .-=  Dat[path][3-lob_num].raw_price_paths[1:s]
        end
        plt = plot!(plt,(0:s-1).*lob_model.Δt,raw_price_paths ,color=5,w=0.6;subplt...,alpha=alpha) ;
        if do_scatter
            plt = scatter!(plt,(0:s-1).*lob_model.Δt,raw_price_paths ,color=5,w=0.6,
                ms=0.5, markerstrokewidth=0.0,  alpha=malpha; subplt...) ;
        end
        
    end
    
    obs_price_paths = Dat[path_to_plot][lob_num].obs_price_paths[1:r]
    if diff
        obs_price_paths .-=  Dat[path_to_plot][3-lob_num].obs_price_paths[1:r]
    end
    
    
    
    plt = plot!(plt,0:r-1, obs_price_paths,color=1,w=2.7;subplt...) ;
    if do_scatter
        plt = scatter!(plt,0:r-1, obs_price_paths,color=1,w=2.7,
            ms=2.5, markerstrokewidth=0.0,  ma=1; subplt...) ;
    end
    
    
    raw_price_paths = Dat[path_to_plot][lob_num].raw_price_paths[1:s]
    plt = plot!(plt,(0:s-1).*lob_model.Δt,raw_price_paths ,color=1,w=0.9;subplt...) ;
    if do_scatter
        plt = scatter!(plt,(0:s-1).*lob_model.Δt,raw_price_paths ,color=1,w=0.6,
            ms=0.9, markerstrokewidth=0.0,  ma=1; subplt...) ;
    end
    
    if do_extra_bit
        ymin = ylims(plt[subplot])[1]
        ymax = ylims(plt[subplot])[2]
        xmin = xlims(plt[subplot])[1]
        xmax = xlims(plt[subplot])[2]
    
        obs_price_paths = Dat[path_to_plot][lob_num].obs_price_paths[r:r+1]
        plt = plot!(plt,r-1:r, obs_price_paths,color=1,w=2.7;subplt...) ;
        if do_scatter
            plt = scatter!(plt,r-1:r, obs_price_paths,color=1,w=2.7,
                ms=2.5, markerstrokewidth=0.0,  ma=1; subplt...) ;
        end
    
        plt = plot!(plt;xlims=[xmin,xmax],ylims=[ymin,ymax],subplt...)
    end
    
    
    
    plt = plot!(plt,legend=false, ylab=L"Price ( $p(t)$ )", xlab=L"Time ( $t$ )";subplt...) ;   
    plt = plot!(plt;kw_for_plot...,subplt...)
    return plt
end

function plot_price_path(Dat, s, lob_num, diff; 
        path_to_plot = 1,kw_for_plot=(),do_scatter=false,subplot=-1,do_extra_bit=false)
     return plot_price_path(plot(), Dat, s, lob_num, diff; path_to_plot = path_to_plot,kw_for_plot=kw_for_plot,
                                            do_scatter=do_scatter,subplot=subplot,do_extra_bit=do_extra_bit)
end
# -

# # Calculations

## Functions to do the all important price impact calculations
function calculate_price_impacts((steps,volumes,inputs),get_set; measured_slob=1, seed=-1) #lob_sets must have the form ((lob_a1,lob_b1),(lob_a2,lob_b2)) where the inner brackets are meant
                                                                                #to be passed to InteractOrderBooks together
    vol_len = length(volumes)
    input_len = length(inputs)
    ti_len = length(steps)

    mean_price_impacts = ones(Float64,ti_len,vol_len,input_len)
    var_price_impacts  = ones(Float64,ti_len,vol_len,input_len)

    for Ind in 1:input_len 
        p_outer = Progress(vol_len,dt=0.1)
        
        for Volume in 1:vol_len
        #Threads.@threads for Volume in 1:vol_len  
            lob_models, sim_start_time, l  = get_set(volumes[Volume],inputs[Ind])

            #try clear_double_dict(Dat) catch e print("Not initialized") end
            #GC.gc()

            Dat = InteractOrderBooks(lob_models, seed, false) ;

            num_paths = lob_models[1].num_paths
            price_impact = zeros(Float64,ti_len,num_paths)
            
            sub_shift = (steps) .+ sim_start_time
            
            for path_num in 1:num_paths
                currdat = Dat[path_num][measured_slob]
                price_impact[:,path_num] = currdat.raw_price_paths[sub_shift] .- currdat.raw_price_paths[sim_start_time]
            end

            mean_price_impacts[:,Volume,Ind] = mean(price_impact,dims=2)#average across paths
            var_price_impacts[:,Volume,Ind] = var(price_impact,dims=2)#average across paths

            next!(p_outer)

        end
    end

    mean_price_impacts = .- mean_price_impacts;
    
    return (mean_price_impacts,var_price_impacts)
end

# # Saving and loading data

function save_data(dat_list;folder_name,file_name,folder_path="")
    if folder_name == "" || file_name == ""
        @warn "Need to provide both a file name and a folder name"
        return -1
    end
    
    @save string(folder_path,"/",folder_name,"/",file_name,".data") dat_list
    return 0
end

function load_data(folder_name,file_name;folder_path="")
    if folder_name == "" || file_name == ""
        @warn "Need to provide both a file name and a folder name"
        return -1
    end
    
    dat_list = load_object(string(folder_path,"/",folder_name,"/",file_name,".data"))
    return dat_list
end

# # Outer wrappers

function obtain_data_list(lob_models;seeds=-1,do_new=true,save_new=false,folder_name="",file_name="",print_labels=-1,folder_path="")
    
    l = length(lob_models)
    
    dat_list = Vector{   Dict{Int64, Dict{Int64, DataPasser}}    }(undef,l)
    
    progress_bars = Array{Progress}(undef,l)
    total_steps = 0
    for i in 1:l
        slob = lob_models[i]
        num_time_steps = slob.N
        total_steps += floor(Int64, slob.num_paths*num_time_steps/1000)
    end
    
    progress_bar = Progress(total_steps,dt=0.1)
    
    if print_labels == -1
        print_labels = repeat([""],length(lob_models))
    end
    
    if seeds == -1
        seeds = rand(length(lob_models)).*1e10
    end
    
    if do_new 
        #for (lob_model,seed,label) in zip(lob_models,seeds,print_labels)
        #Threads.@threads for (num,lob_model,seed,label,progress_bar) = zip(1:l,lob_models,seeds,print_labels,progress_bars)
        
        for i in 1:length(lob_models)
        #Threads.@threads for i = (1:length(lob_models))
            lob_model = lob_models[i]
            seed = seeds[i]
            label = print_labels[i]
            
            print(label)
            dat_list[i] =  InteractOrderBooks([lob_model], seed, true;)
        end

        if save_new
            save_data(dat_list;folder_name=folder_name,file_name=file_name,folder_path=folder_path)
            print("Saved successfully\n")
        end
            
    else
            
        dat_list =  load_data(folder_name,file_name,folder_path=folder_path)
        print("Loaded successfully\n")
            
    end
        
    return dat_list
end

# +
function obtain_price_impacts((steps,volumes,combined),get_set;do_new=true,save_new=false,folder_name="",file_name="",folder_path="")
    if do_new 
        (mean_price_impacts,var_price_impacts) = calculate_price_impacts((steps, volumes,  combined), get_set)

        if save_new
            save_data((mean_price_impacts,var_price_impacts);folder_name=folder_name,file_name=file_name,folder_path=folder_path)
            print("Saved successfully\n")
        end
            
    else
            
        (mean_price_impacts,var_price_impacts) =  load_data(folder_name,file_name,folder_path=folder_path)
        print("Loaded successfully\n")
            
    end
        
    return (mean_price_impacts,var_price_impacts)

end

# +
"""
    quick_plot(lob_models::Array{SLOB},sim_start_time::Int64,how_many::Int64=12; <keyword args>)

A function which, given an array of LOB models, `lob_models`, will run the model using `InteractOrderBooks` and then 
call `plot_density_visual` `how_many` times in order to display the results. This displays `how_many` plots starting at 
`sim_start_time` and ending at `sim_start_time + how_many`. 

# Arguments
- `Dat::()=-1`: If the results of calling `InteractOrderBooks` on `lob_models` have already been computed, they may 
                be passed in here to skip redoing that computation.
- `lob_num::Int64=1`: Which of the LOBs to display results from
- `seed::Int64=-1`: Which seed to pass to `InteractOrderBooks` when running the simulation.
- `size::Tuple(Int64,Int64)`: How large the plots containing all the computed plots should be.
- `for_visual::Tuple()`:A tuple with keywords which will be passed to the `plot_density_visual` function as `plot_density_visual(;for_visual...)`. 
                         Thus one may do, e.g: `for_visual=(x_axis_width=10,)` which results in `plot_density_visual(;x_axis_width=10)` via the `...` operation.
"""
function quick_plot(lob_models,RLParams,sim_start_time,how_many=12;for_visual=(),Dat=-1,lob_num=1,seed=-1,size=(300,300))
    if Dat == -1 
        if length(RLParams)==0
            Dat = InteractOrderBooks(lob_models, seed, true) ;
        else
            Dat = InteractOrderBooks(lob_models, RLParams, seed, true) ;
        end
            
    end
    
    p_arr1 = Array{Plots.Plot{Plots.GRBackend},1}(undef,how_many)
    
    for i in [1:how_many;]
        p_arr1[i] = plot_density_visual(Dat,sim_start_time-2+i,lob_num;for_visual...)
        
    end
    
    (l,dim) = get_layout_for_n_plots(how_many,size)

    plot(p_arr1...,layout = l,size=dim)
    
    return (Dat,p_arr1)
end


# -

# # Junk

# +
# function obtain_price_impacts_mod(volumes,combined,steps,get_set;do_new=true,save_new=false,folder_name="",file_name="",folder_path="")
    
    
#     if do_new 
#         (mean_price_impacts,var_price_impacts,l_values) = calculate_price_impacts_mod(volumes,  combined,   get_set, steps)

#         if save_new
#             save_data((mean_price_impacts,var_price_impacts,l_values);folder_name=folder_name,file_name=file_name,folder_path=folder_path)
#             print("Saved successfully\n")
#         end
            
#     else
            
#         (mean_price_impacts,var_price_impacts,l_values) =  load_data(folder_name,file_name,folder_path=folder_path)
#         print("Loaded successfully\n")
            
#     end
        
#     return (mean_price_impacts,var_price_impacts,l_values)

# end

# +
# ## Functions to do the all important price impact calculations
# function calculate_price_impacts_mod(volumes,inputs,get_set,steps; measured_slob=1, seed=-1) #lob_sets must have the form ((lob_a1,lob_b1),(lob_a2,lob_b2)) where the inner brackets are meant
#                                                                                 #to be passed to InteractOrderBooks together
#     vol_len = length(volumes)
#     input_len = length(inputs)

#     mean_price_impacts = ones(Float64,steps,vol_len,input_len)
#     var_price_impacts  = ones(Float64,steps,vol_len,input_len)
#     l_values = ones(Int64,steps)
    
#     sub = 1:steps

#     for Ind in 1:input_len 
#         p_outer = Progress(vol_len,dt=0.1)
        
#         for Volume in 1:vol_len
#         #Threads.@threads for Volume in 1:vol_len  
#             lob_models, sim_start_time, l  = get_set(volumes[Volume],inputs[Ind])

#             #try clear_double_dict(Dat) catch e print("Not initialized") end
#             #GC.gc()

#             Dat = InteractOrderBooks(lob_models, seed, false) ;

#             num_paths = lob_models[1].num_paths
#             price_impact = zeros(Float64,steps,num_paths)
            
#             sub_shift = (sub) .+ sim_start_time
            
#             for path_num in 1:num_paths
#                 currdat = Dat[path_num][measured_slob]
#                 price_impact[sub,path_num] = currdat.raw_price_paths[sub_shift] .- currdat.raw_price_paths[sim_start_time]
#             end

#             mean_price_impacts[sub,Volume,Ind] = mean(price_impact,dims=2)
#             var_price_impacts[sub,Volume,Ind] = var(price_impact,dims=2)
#             l_values[Ind] = l

#             next!(p_outer)

#         end
#     end

#     mean_price_impacts = .- mean_price_impacts;
    
#     return (mean_price_impacts,var_price_impacts,l_values)
# end

# +
# function obtain_price_impacts(volumes,combined,get_set;do_new=true,save_new=false,folder_name="",file_name="",folder_path="")
    
    
#     if do_new 
#         (mean_price_impacts,var_price_impacts) = calculate_price_impacts(volumes,  combined,   get_set)

#         if save_new
#             save_data((mean_price_impacts,var_price_impacts);folder_name=folder_name,file_name=file_name,folder_path=folder_path)
#             print("Saved successfully\n")
#         end
            
#     else
            
#         (mean_price_impacts,var_price_impacts) =  load_data(folder_name,file_name,folder_path=folder_path)
#         print("Loaded successfully\n")
            
#     end
        
#     return (mean_price_impacts,var_price_impacts)

# end

# +
## Functions to do the all important price impact calculations
# function calculate_price_impacts(volumes,inputs,get_set; measured_slob=1, seed=-1) #lob_sets must have the form ((lob_a1,lob_b1),(lob_a2,lob_b2)) where the inner brackets are meant
#                                                                                 #to be passed to InteractOrderBooks together
#     vol_len = length(volumes)
#     input_len = length(inputs)

#     mean_price_impacts = ones(Float64,vol_len,input_len)
#     var_price_impacts  = ones(Float64,vol_len,input_len)

#     for Ind in 1:input_len 
#         p_outer = Progress(vol_len,dt=0.1)
        
#         for Volume in 1:vol_len
#         #Threads.@threads for Volume in 1:vol_len  
#             lob_models, sim_start_time, l  = get_set(volumes[Volume],inputs[Ind])

#             #try clear_double_dict(Dat) catch e print("Not initialized") end
#             #GC.gc()

#             Dat = InteractOrderBooks(lob_models, seed, false) ;

#             num_paths = lob_models[1].num_paths
#             price_impact = zeros(Float64,num_paths)
#             for path_num in 1:num_paths
#                 price_impact[path_num] = Dat[path_num][measured_slob].raw_price_paths[sim_start_time+l] - Dat[path_num][measured_slob].raw_price_paths[sim_start_time]
#             end

#             mean_price_impacts[Volume,Ind] = mean(price_impact)
#             var_price_impacts[Volume,Ind] = var(price_impact)

#             next!(p_outer)

#         end
#     end

#     mean_price_impacts = .- mean_price_impacts;
    
#     return (mean_price_impacts,var_price_impacts)
# end

# +
# """
#     fit_and_plot_price_impact( volumes::Array{Float64}, mean_price_impacts::Array{Float64}, var_price_impacts::Array{Float64}, labels::Array{String}; <keyword arguments>)

# Given the results of calling the function `calculate_price_impacts` which returns the `mean_price_impacts` and `var_price_impacts` for each of the `volumes` used, pass them to this function
#  to have them plotted. When plotting them, give them custom labels to go in the legend by passing `labels`.

# # Keyword arguments
# - `sub::Array{Float64}=-1`: Which subset of the volumes to plot. Will be run as `volumes = volumes[sub]`. Allows for trying different subsets quickly.
# - `do_kinks::Array{Float64}=true`: Whether to highlight kinks on the plot by using arrows.
# - `colors::Array{Float64}=-1`: What colors to assign to each line in the order that they are drawn.
# - `do_power_fit::Bool=false`: Whether to do a fit using a function of the form "y = ax^b" and include it on the plot.
# - `xticks::Array{Float64}=-1`: Which 'xtick' positions to highlight. Will be passed to `plot` as `plot(;xticks=xticks)`.
# - `new_plot::Bool=false=-1`: Whether to create a new plot or draw to an existing one.
# - `for_plot::Tuple(...)=()`:
# - `do_ribbon::Bool=false`:
# - `do_horiz_log_shift::Bool=true`:
# - `do_log_fit::Bool=true`:
# - `do_log_plot::Bool=false`:
# - `do_vert_log_shift::Bool=true`:
# - `use_only_straight_region::Bool=false`:
# - `shift_y_ticks::Bool=false`:
# - `do_just::Int64=-1`:
# - `straight_line_frac_to_plot::Float64=-1`:
# - `square_the_price::Bool=false`:
# """

# function fit_and_plot_price_impact(volumes,mean_price_impacts,var_price_impacts,labels;
#                                                 sub=-1,do_kinks=true,colors=-1,do_power_fit=false,xticks=-1,new_plot=false,forplot=(),
#                                                 do_ribbon=true,do_horiz_log_shift=true,do_log_fit=true,do_log_plot=false,
#                                                 do_vert_log_shift=true,use_only_straight_region=false,shift_y_ticks=false,do_just=-1,
#                                                 straight_line_frac_to_plot=-1,square_the_price=false,
#                                                 modify_input_v=identity,modify_input_p=identity,
#                                                 modify_plot_v=identity, modify_plot_p=identity)
#     # the below is just for readability
    
#     modify_plot__v = modify_plot_v
#     modify_plot__p = modify_plot_p
    
#     if new_plot #want a new plot?
#         plot()
#     end
    
    
    
#     # Initialize stuff
#     #########################################################################
#     li_len = length(labels)
#     vi_len = length(volumes)
    
#     if typeof(do_power_fit)==Bool
#         do_power_fit = repeat([do_power_fit],li_len)
#     end
    
#     if typeof(do_log_fit)==Bool
#         do_log_fit = repeat([do_log_fit],li_len)
#     end
    
#     if do_just==-1
#         do_just = repeat([true],li_len)
#     end
    
#     if sub==-1
#         sub = 1:vi_len
#     end
    
#     if straight_line_frac_to_plot == -1
#         straight_line_frac_to_plot = repeat([1.0],li_len)
#     end
    
#     if colors==-1 #change later!
#         if li_len<=5
#             colors = ["blue","red","green","orange","purple"]
#         else
#             colors = 1:li_len
#         end
#     end
    
#     shift = do_horiz_log_shift ? 1 : 0
    
#     if do_log_plot  && (modify_plot__v == identity)
#     #    modify_plot__v = (v) -> log(v+1)
#     end
#     #########################################################################
    
#     volumes_input = modify_input_v.(volumes[sub]) #the original volumes given before manipulation
#     volumes_plot  = modify_plot__v.(volumes[sub])#log.(volumes.+1)[:]
    
#     for ind in 1:li_len
        
#         if !do_just[ind] 
#             continue
#         end
        
#         mean_price_input    =  modify_input_p.(mean_price_impacts[sub,ind])
#         mean_price_plot     =  modify_plot__p.(mean_price_impacts[sub,ind])#mean_price_impacts

#         var_price_input     =  modify_input_p.( var_price_impacts[sub,ind])
#         var_price_plot      =  modify_plot__p.( var_price_impacts[sub,ind])#var_price_impacts
        
#         # the below sets the upper cutoff to either the highest volume or the point with the largest derivative
#         upper_fit_cut_off = length(sub) #upper cut off is highest volume
#         if use_only_straight_region #upper cutoff is point with largest derivative
#             upper_fit_cut_off = findmax(diff(mean_price_input))[2]
#             upper_fit_cut_off = upper_fit_cut_off == nothing ? 0 : upper_fit_cut_off
#         end
#         sub_to_fit = sub[1:upper_fit_cut_off]
        
        
#         # the below fits a log function to the input data
#         if do_log_fit[ind]
#             a,b = log_fit( volumes_input[sub_to_fit].+shift ,  mean_price_input[sub_to_fit] )
#             a = do_vert_log_shift ? a : 0

            
#             line_plot_cut_off = straight_line_frac_to_plot[ind] * upper_fit_cut_off
#             line_plot_cut_off = min( floor(Int64,line_plot_cut_off) ,  length(sub))
#             sub_to_plot = sub[1:line_plot_cut_off]
            
#             y_line_log_fit = a.+b.*log.(volumes_input[sub_to_plot].+shift)
            
#             plot!(         volumes_plot[sub_to_plot] , modify_plot__p.(y_line_log_fit) ,
#                            label=string("Log fit: ",round(a,digits=2)," + ",round(b,digits=2),"log(V/A+1)"),w=1.5,color=colors[ind])
            
#         end
        
#         # the below fits a power function to the input data
#         if do_power_fit[ind]
#             c,d = power_fit(volumes_input[sub_to_fit],mean_price_input[sub_to_fit])
            
#             y_line_power_fit = c.*(volumes_input[sub_to_fit]).^d
            
#             plot!(volumes_plot[sub_to_fit], modify_plot__p.(y_line_power_fit),
#                 label=string("Power fit: ",round(c,digits=2),"(V/A)^",round(d,digits=2)),
#                         w=1.5,color=colors[ind],linestyle=:dash)
#         end
        
#         # plots the actual price impact points
#         scatter!(      volumes_plot,   mean_price_plot,
#                 label=labels[ind],ms=1.5,markerstrokewidth=0.1,  ma=1,color=colors[ind])
        
#         # plots a ribbon of uncertainty around the price impact points
#         if do_ribbon
#             plot!(     volumes_plot,   mean_price_plot,  ribbon=var_price_plot.^0.5,alpha=0,
#                 fillalpha=0.4,fillcolor=colors[ind],label="")
#         end
            
#         # if xticks not provided, use global xticks??????
#         if xticks!=-1
#             plot!(xticks=xticks)
#         end


#         # detect the kinks in the graph
#         if do_kinks
            
#             vol_scale = (volumes_input[end]-volumes_input[1])/20
#             impact_scale = (maximum(mean_price_input,dims=1)[1]-minimum(mean_price_input,dims=1)[1])/30
            
#             second_deriv = get_second_derivative(volumes_input,mean_price_input)
#             kink_position = findnext(x->x>0,second_deriv,1)
            
#             kink_counter = 1
            
#             while !(kink_position===nothing) && kink_counter<=20
#                 target_x, target_y = (volumes_input[kink_position+1],mean_price_impacts[kink_position+1])
                
#                 quiver!([target_x+vol_scale],[target_y-impact_scale],quiver=([-vol_scale],[impact_scale]),color=colors[ind])
#                 scatter!([target_x],[target_y],markershape=:star5,color=colors[ind],
#                     label=string("Kink at position ",round(target_x,digits=2)),markerstrokewidth=0.1,  ma=1)
                
#                 kink_position = findnext(x->x>0,second_deriv,kink_position+2)
                
#                 kink_counter += 1
#             end
#         end

#     end
    
#     # label the xaxis
#     my_xlabel = do_log_plot ? "log(Volume)" : "Volume"
#     plot!(xlabel=my_xlabel,ylabel="Price impact i.e. p(t+1)-p(t)";forplot...)
# end
