# -*- coding: utf-8 -*-
# # Setup

import StatsBase:autocor
using Plots
using Statistics
using Distributions
using StatsPlots
using StatsBase
using Plots.PlotMeasures
using Extremes
using CurveFit

mutable struct StylizedFactsPlot
    price_path::Matrix{Float64}
    log_price_path::Matrix{Float64}
    log_returns::Matrix{Float64}
    order_signs::Matrix{Float64}
    order_flow_acf::Matrix{Float64}
    log_returns_acf::Matrix{Float64}
    abs_log_returns_acf::Matrix{Float64}
    N::Int
    L::Int
    M::Int
    l_conf::Float64
    u_conf::Float64
    dats::Array{DataPasser}
end

# +
function StylizedFactsPlot(dat_arr::Array{Dict{Int64, Dict{Int64, DataPasser}}, 1};do_raw=true, real_upper_cut_off = -1)
    len = length(dat_arr)
    
    Nmin = Inf 
    Mmin = Inf
    
    for i in 1:len
        if do_raw
            example_data = dat_arr[i][1][1].raw_price_paths[1:end-1,1]
        else
            example_data = dat_arr[i][1][1].obs_price_paths[1:end-1,1]
            
            if real_upper_cut_off != -1
                sub = 1:floor(Int64,real_upper_cut_off)
                example_data = example_data[sub]
            end
            
        end
        N = length(example_data)
        if N < Nmin
            Nmin = N
            Mmin = length(autocor(example_data))
        end
    end
    
    N = Nmin
    M = Mmin
    
    
        # Should use: ceil(Int64,min(N-1, 10*log10(N)))+1  to determine M but doesn't seem to work nicely
        # the above is the length of the array returned by "autocor"
        # is a fudge. Will probably break later. If you see array size errors, its probably this
    
    price_path =          zeros(Float64,len,N)
    log_price_path =      zeros(Float64,len,N)
    log_returns =         zeros(Float64,len,N-1)
    order_signs =         zeros(Float64,len,N)
    order_flow_acf =      zeros(Float64,len,M)
    log_returns_acf =     zeros(Float64,len,M)
    abs_log_returns_acf = zeros(Float64,len,M)
    dats                = Array{DataPasser}(undef,len)
    
    
    for i in 1:len
        # the keyword raw below is used to distuinguish the arrays from the temporary variables
        if do_raw
            raw_price_path           = dat_arr[i][1][1].raw_price_paths[1:N,1]#[1:end-1,1]
        else
            raw_price_path           = dat_arr[i][1][1].obs_price_paths[1:N,1]#[1:end-1,1]
        end
        
        raw_log_price_path       = raw_price_path
        raw_log_returns          = diff(raw_price_path)
        raw_tick_rule            = tick_rule(raw_price_path)
        print("Changed")
        
        # the raw values are then stored in arrays
        price_path[i,:]          = raw_price_path
        log_price_path[i,:]      = raw_log_price_path  
        log_returns[i,:]         = diff(raw_log_price_path) # returns log_returns[i+1] - log_returns[i] for all i in 1 to end-1
        order_signs[i,:]         = raw_tick_rule
        order_flow_acf[i,:]      = autocor(raw_tick_rule)
        log_returns_acf[i,:]     = autocor(raw_log_returns)
        abs_log_returns_acf[i,:] = autocor(abs.(raw_log_returns))
        dats[i]                  = dat_arr[i][1][1]
    end

    l_conf = -1.96 / sqrt(N) #lower confidence
    u_conf = +1.96 / sqrt(N) #upper confidence

    return StylizedFactsPlot(price_path, log_price_path, log_returns,
        order_signs, order_flow_acf, log_returns_acf, abs_log_returns_acf,
        N, len, M, l_conf, u_conf, dats) 
    
end
# -

# # Different plots

# ## Midprice stuff

function tick_rule(mid_prices)
    # Classify trades as buyer or seller initiated according to
    # the tick rule. Using the previous mid-price, if the transaction
    # price is higher then the trade is classified as buyer-initiated
    # if the transaction price is lower then the trade is classified as
    # seller initiated. If there is no price change, but the previous
    # tick change was up then the trade is a buy else it is a sell.
    # We cannot classify the first trade.

    N = size(mid_prices,1)
    signs = zeros(Int, N)
    previous_tick_change = 0
    for i in 2:N
        if mid_prices[i] == mid_prices[i-1]
            if previous_tick_change > 0
                signs[i] = 1
            elseif previous_tick_change < 0
                signs[i] = -1
            end

        elseif mid_prices[i] > mid_prices[i-1]
            signs[i] = 1
            previous_tick_change = 1
        elseif mid_prices[i] < mid_prices[i-1]
            signs[i] = -1
            previous_tick_change = -1
        end
    end
    return signs
end

function plot_mid_price_path(sf::StylizedFactsPlot; title="Mid-Price Path", alpha=1, 
        legend_pos=(:right,:top), frac_choice1=(1,0), frac_choice2=(1,0),
        create_space=0, zoom=3)
    
    p = plot(title=title, xlab=L"Trade event count ( $\ell$ )", ylab=L"Mid-Price ( $p_\ell$ )")
    
    frac = 0.3
    xb = 0.05
    yb = 0.05
    
    legend_x = legend_pos[1]
    legend_y = legend_pos[2]
    
    p = plot!(p, inset = (1,bbox(xb, yb, frac, frac, legend_y, legend_x))  )
    target_time = zoom 
     
    
    
    
    for i in 1:sf.L
        p = plot!(p, sf.price_path[i,:], legend=false, color=i,alpha=alpha);
    end
    
    
    
    
    
    
    flip_x = (legend_x == :right) ? (f) -> 1-f : (f) -> f
    flip_y = (legend_y == :top) ? (f) -> 1-f : (f) -> f
    
    if create_space > 0
        b = ylims(p[1])[1]
        t = ylims(p[1])[2]
        p = plot!(p, ylims=[b,b+(1/(1-frac*create_space))*(t-b)],subplot=1)
    end
    
    if create_space < 0
        b = ylims(p[1])[1]
        t = ylims(p[1])[2]
        p = plot!(p, ylims=[t-(1/(1-frac*create_space))*(t-b),t],subplot=1)
    end
    
    
    if legend_x == :left
      #p = plot!(p,xmirror=true,subplot=2)
      #p = plot!(p,xmirror=false,subplot=1)
    end
    
    if legend_y == :top
       #p = plot!(p,ymirror=true,subplot=2)
       #p = plot!(p,ymirror=false,subplot=1)
    end
    
   p = plot!(p,mirror=true;subplot=2)
    
    p = plot!(p,frame=:box,subplot=2)
    
    
    #cum_t = sf.dats[1].slob.Δts_cum
    #N = sum(cum_t .<= target_time)
    #t_plot = cum_t[1:N]
    #sub = 1:N
        
    
    for i in 1:sf.L
        slob = sf.dats[i].slob
        cum_t = slob.Δts_cum
       
        sub = cum_t .<= target_time
        t_plot = cum_t[sub]
       
        p_plot = sf.dats[i].raw_price_paths[sub]
        
        mf = (x) -> floor(Int64,x)
        mf2 = (x) -> !iszero(x)
        sub2 = (mf2.(vcat(diff(mf.(t_plot)),0)))
        print(sub2)
       
        p = plot!(   p, t_plot, p_plot , legend=:none,
            subplot=2, color = i)
        p = scatter!(p, t_plot, p_plot , legend=:none,
            ms=1,markerstrokewidth=0.1,  ma=1,
            subplot=2, color = i)
        p = scatter!(p, t_plot[sub2], p_plot[sub2], legend=:none,
            ms=3,markerstrokewidth=0.1,  ma=1,
            subplot=2, color = i)
    end
    
    # change y ticks of inset to not be horrible
   
    b = ylims(p[2])[1]
    t = ylims(p[2])[2]
    p = plot!(p,yticks=[b+0.1(t-b),t-0.1(t-b)],subplot=2)
    p = plot!(p,xticks=range(0,target_time,step=1),subplot=2)
    
    # draw arrow to inset region
    p = arrow_from_abs_to_frac(p,[0,sf.price_path[1,1]],[flip_x(xb + frac_choice1[1]*frac),flip_y(yb + frac_choice1[2]*frac)];subplot=1,include_dot=true)
    p = arrow_from_abs_to_frac(p,[0,sf.price_path[1,1]],[flip_x(xb + frac_choice2[1]*frac),flip_y(yb + frac_choice2[2]*frac)];subplot=1,include_dot=true)
    
    return p
end

# ## Returns stuff

function plot_log_returns(sf::StylizedFactsPlot; title="Log Returns",alpha=-1)
    
    if alpha==-1
        alpha = 1/sf.L
    end
    
    
    variances = [var(sf.log_returns[i,:]) for i in 1:sf.L]
    my_permutation = sortperm(variances)[end:-1:1]
    ordered_by_variances = sortslices(sf.log_returns,dims=1,by=(x)->var(x),rev=true)
    
    plot(title = title, xlab=L"Trade event count ( $\ell$ )", ylab=L"Log Returns ( $p(\ell+1)-p(\ell)$ )")
    for i in 1:sf.L
        plot!(ordered_by_variances[i,:], legend=false, alpha=alpha, color=my_permutation[i]);
    end
    return plot!()
end

# ## Histogram stuff

function plot_hist_log_returns(sf::StylizedFactsPlot; title="Log Returns Histogram",alpha=1)
    plot(title=title,xlab="Density", ylab=L"Log Returns ( $p(\ell+1)-p(\ell)$ )",legend=:none)
    
    y_global_min = +100000
    y_global_max = -100000
    x_global_max = -100000
    
    for i in 1:sf.L
        data = sf.log_returns[i,:]
        
        y_min = minimum(data)
        y_max = maximum(data)
        
        x_range = range(y_min,stop = y_max,length=200)
        stephist!(data , normalize=:pdf, color=i, alpha=alpha,direction=:x)         #, label="observed returns");
        
        fitted_normal = Distributions.fit(Normal, data)
        f(x) = pdf(fitted_normal, x)
        plot!(f.(x_range),x_range,color=i,alpha=alpha)#, label="Normal Distribution");
        
        x_max = maximum(f.(x_range))
        
        y_global_min = minimum([y_min, y_global_min])
        y_global_max = maximum([y_max, y_global_max])
        x_global_max = maximum([x_max, x_global_max])
    end
    
    plot!(ylim=[y_global_min,y_global_max],xlim=[0,x_global_max])
    return plot!()
end

# ## QQ stuff

function plot_qq_log_returns(sf::StylizedFactsPlot; title="Log Returns Normal Q-Q Plot",alpha=-1)
    
    if alpha==-1
        alpha = 1/(sf.L)
    end
    
    plot(title=title,xlab="Theoretical Quantiles", ylab="Sample Quantiles")
    for i in 1:sf.L
        data = sf.log_returns[i,:]
        sorted_data = sort(data)                         #sort the data
        
        fitted_normal = StatsBase.fit(Normal,sorted_data)          #fit normal to the data
        x_axis_inner = [i/(sf.N+1) for i in 1:sf.N]      #create a distribution of percentiles that each of the ordered data points should lie at
        
        q(x)      = quantile(fitted_normal,x)            #get the quantile of that percentile
        x_axis    = q.(x_axis_inner)                     #get the x axis points at which to plot the data
        
        
        #f(x) = pdf(fitted_normal,x)
        #alpha_arr = f.(x_axis) .* (-1) .+ f(x_axis[floor(Int64,sf.N/2)])
        #lowest = 0.015
        #highest = 0.5
        #alpha_arr = (alpha_arr./maximum(alpha_arr)).^12 .* (highest-lowest) .+ lowest #gives a range [lowest,highest]
        #alpha_arr = vcat(alpha_arr,[alpha_arr[end]])
        #return plot(alpha_arr)
        
        #qqplot!(fitted_normal, data,        color=repeat([4],sf.N), 
        #                                    markersize=repeat([3.0],sf.N), markerstrokewidth=repeat([0.0],sf.N), 
        #                                    alpha=repeat([0.5],sf.N), markershape=:star5)
        
        
        scatter!(x_axis,       sorted_data, color=repeat([i],sf.N), 
                                            markersize=repeat([0.8],sf.N), markerstrokewidth=repeat([0.0],sf.N),label=string("Data ",i)) 
                                            #alpha=alpha_arr)
       
        #if i==2
        plot!(x_axis,x_axis,color=i,label="Expected")
        #end
    end
    
    
    
    return plot!(legend=:none)
end

# ## ACF plots

# +
function plot_acf_log_returns(tar_plot,sf::StylizedFactsPlot; title="Log Returns Autocorrelation",alpha=1,for_plot=())
    return plot_an_acf(tar_plot,sf, sf.log_returns_acf; title=title, alpha=alpha, option=3, for_plot)
end

function plot_acf_log_returns(sf::StylizedFactsPlot; title="Log Returns Autocorrelation",alpha=1,for_plot=())
    return plot_an_acf(plot(),sf, sf.log_returns_acf; title=title, alpha=alpha, option=3, for_plot)
end

# +
function plot_acf_order_flow(tar_plot, sf::StylizedFactsPlot; title="Order Flow Autocorrelation (Tick Rule)",alpha=1,for_plot=())
    return plot_an_acf(tar_plot, sf, sf.order_flow_acf; title=title, alpha=alpha, option=3, for_plot)
end

function plot_acf_order_flow(sf::StylizedFactsPlot; title="Order Flow Autocorrelation (Tick Rule)",alpha=1,for_plot=())
    return plot_an_acf(plot(),sf, sf.order_flow_acf; title=title, alpha=alpha, option=3, for_plot)
end

# +
function plot_acf_abs_log_returns(tar_plot, sf::StylizedFactsPlot; title="Absolute Log Returns Autocorrelation",alpha=1,for_plot=())
    return plot_an_acf(tar_plot, sf, sf.abs_log_returns_acf; title=title, alpha=alpha, option=3,for_plot)
end

function plot_acf_abs_log_returns(sf::StylizedFactsPlot; title="Absolute Log Returns Autocorrelation",alpha=1,for_plot=())
    return plot_an_acf(plot(), sf, sf.abs_log_returns_acf; title=title, alpha=alpha, option=3,for_plot)
end

# +
function plot_an_acf(tar_plot,sf::StylizedFactsPlot,acf_values::Matrix{Float64}; title="",alpha=1,option=1,do_log=false,for_plot=())
    
    if do_log
        maybe_log = (x) -> -1 * sign(x) * log(abs(x))
        my_label = "-log(|ACF|)*sign(ACF)"
    else 
        maybe_log = (x) -> (x)
        my_label = "ACF"
    end
    
    sub_plot = plot!(tar_plot, title=title, xlab=L"Lag in trade events ( $\ell$ )",ylab=my_label; for_plot...)
    
    for i in 1:sf.L
        x_axis_pos = collect(1:sf.M).+(i-1)/sf.L*0.6
           
        plot!( sub_plot,  x_axis_pos, maybe_log.(acf_values[i,:]), color=i, alpha=alpha, seriestype=:sticks; for_plot...)
        scatter!(x_axis_pos, maybe_log.(acf_values[i,:]), color=i, alpha=alpha, markersize=2.0, markerstrokewidth=0.0;for_plot...)
    end
    
    if !do_log
        plot!(x -> 0, linestyle=:solid, color="black"; for_plot...)
    end
    
    plot!(x -> maybe_log(sf.l_conf), color="grey", alpha=0.5; for_plot...)
    plot!(x -> maybe_log(sf.u_conf), color="grey", alpha=0.5; for_plot...)

    return sub_plot
end

function plot_an_acf(sf::StylizedFactsPlot,acf_values::Matrix{Float64}; title="",alpha=1,option=1,do_log=false,for_plot=())
    return plot_an_acf(plot(),sf,acf_values;title = title,alpha=alpha,option=option,do_log=do_log,for_plot=for_plot)
end

# +
function plot_indented_acfs(plt,sf)
    my_title = (title="",)
    p = plot_acf_order_flow(plt,sf;for_plot=(ylab=L"ACF Order flow ( $\ell$ )",),my_title...)
    frac = 0.8
    p = plot!(p, inset = (1,bbox(0.0, 0.0, frac, frac, :top, :right))  )
    p = plot_acf_log_returns(p,sf;for_plot=(subplot=2,title="",ylab=L"ACF Log Returns ( $\ell$ )",xlab=""), my_title...)
    frac = 0.75
    p = plot!(p, inset = (2,bbox(0.0, 0.0, frac, frac, :top, :right))  )
    p = plot_acf_abs_log_returns(p,sf;for_plot=(subplot=3,title="",ylab=L"ACF abs. Log Returns ( $\ell$ )",xlab=""), my_title...)
    return plot!(legend=:none)
end

function plot_indented_acfs(sf)
    return plot_indented_acfs(plot(),sf)
end
# -

# ## Exceedance stuff

# +
function gp_qq_plot(plt,fm;plot_args=())
    all_plot_data = qqplot_data(fm)
    x_axis = all_plot_data[:,1]
    y_data_points = all_plot_data[:,2]

    scatter!(plt,x_axis,y_data_points, markersize=2.0, markerstrokewidth=0.0,label="Data";plot_args...)
    return plot!(plt,x_axis,x_axis, markersize=2.0, markerstrokewidth=0.0,label="Fitted GP";plot_args...)
end

gp_qq_plot(fm) = gp_qq_plot(plot(),fm)

function gp_density_plot(plt,fm;do_scatter=false,no_bins=10,plot_args=())
    all_plot_data = histplot_data(fm)[:d]
    
    exceedances = fm.model.data.value
    exc_min = minimum(exceedances)
    exc_max = maximum(exceedances)
    my_bins = range(exc_min, exc_max, length=no_bins+1)
    
    y_vals_fit = all_plot_data[:,2]
    x_axis = range(exc_min,exc_max,length=length(y_vals_fit))

    if do_scatter
        scatterhist!(plt,exceedances,bins=my_bins,normalize=:pdf,label="Data";plot_args...)
    else
        stephist!(plt,exceedances,bins=my_bins,normalize=:pdf,label="Data";plot_args...)
    end
    return plot!(plt,x_axis,y_vals_fit, color="red",label="Fitted GP";plot_args...)
end

gp_density_plot(fm) = gp_density_plot(plot(),fm)


function gp_probability_plot(plt,fm;plot_args=())
    all_plot_data = probplot_data(fm)
    x_axis = all_plot_data[:,1]
    y_axis_points = all_plot_data[:,2]

    scatter!(plt,x_axis,y_axis_points, markersize=2.0, markerstrokewidth=0.0,label="Data";plot_args...)
    return plot!(plt,[0,1],[0,1],color="red",label="Fitted GP";plot_args...)
end

gp_probability_plot(fm) = gp_probability_plot(plot(),fm)

function gp_return_level_plot(plt,fm;plot_args=())
    all_plot_data = returnlevelplot_data(fm)
    x_axis = log10.(all_plot_data[:,2])
    y_data_points = all_plot_data[:,1]
    y_fit = all_plot_data[:,3]

    scatter!(plt,x_axis ,y_data_points ,markersize=2.0, markerstrokewidth=0.0,label="Data";plot_args...)
    max_y = maximum(y_data_points)
    return plot!(plt,x_axis ,y_fit, label="Fitted GP";plot_args...)
end

function gp_return_level_plot(fm;plot_args=())
    return gp_return_level_plot(plot(),fm;plot_args=plot_args)
end
# -

function create_space(amount,side;subplot=1)
    # future work
    if side == :top
        h = ylims(plot!())[2]
        plot!(ylims=[0,(1/(1-frac*1.26))*h],subplot=1,xlabel=L"Return Period ( $1/p$ )",ylabel="Return Level")
    end
end

# +
function plot_exceedance_plot(plt,sf;tolerance=1.0)
    
    #mydata1 = sort(sf.price_path[1,:])
    #mydata2 = sort(sf.price_path[2,:])
    #mydata3 = sort(sf.price_path[3,:])

    #mydata = [mydata1,mydata2,mydata3]
    #mydata = [sf.price_path[i,:] for i in 1:sf.L]
    mydata = [sf.log_returns[i,:] for i in 1:sf.L]
    
    if length(tolerance) == 1
        tolerance = repeat([tolerance],sf.L)
    end
    
    frac = 0.2
    left = 0.13
    step = 0.3
    #plot!(inset = (1,bbox(0.0, b, frac, frac, :bottom, :right))  )
    #plot!(inset = (1,bbox(0.0, 8*b, frac, frac, :bottom, :right))  )
    #plot!(inset = (1,bbox(0.35, b+0.3*0+b*0, frac, frac, :bottom, :right))  )
    
    plot!(inset = (1,bbox(left, 0, frac, frac))  )
    plot!(inset = (1,bbox(left+step, 0, frac, frac)) )
    
    if sf.L == 3
        plot!(inset = (1,bbox(left+step+step, 0, frac, frac))   )
    end
    

    for i in 1:sf.L
        # mrl_prop creates its own set of x values. This returns the index into these values
        # above a which a straight line fits as pos, 
        # as well as the set of x values itself
        (pos,x_vals) = find_mrl_prop_to_achieve(mydata[i];tolerance=tolerance[i])  
        threshold = x_vals[pos] # the actual amount which needs to be exceeded
        l = length(x_vals)

        # do main return level plot
        exceedances = get_exceedances(mydata[i],threshold)
        fm = gpfit(exceedances)
        gp_return_level_plot(plot!(),fm;plot_args=(color=i,))
        
        

        # pos is the first position above which a straight line fits, but we want to show a couple below that
        
        lower_pos = max(l - 2 * (l - pos),1)

        # plot the straight line
        plot_args_ = (subplot=i+1,legend=:none,color=i)
        my_mrl_plot(mydata[i];plot_args=plot_args_,above=lower_pos)
        
        # plot the points
        plot_args_ = (subplot=i+1,color=i)
        my_mrl_plot(mydata[i];plot_args=plot_args_,do_linear_fit=true,do_dots=false,above=pos)

        # plot the ticks at the lower_pos, the threshold itself and the upper value
        my_x_ticks = (x->round(x,digits=2)).([x_vals[lower_pos],x_vals[pos],x_vals[l]])
        plot!(xticks = my_x_ticks,subplot=i+1)
        my_y_lab = (i==1) ? ("Mean excess") : ("")
        my_x_lab = (true) ? (L"\mu") : ("")
        plot!(ylab=my_y_lab,xlab=my_x_lab,subplot=i+1)

        # plot the threshold itself with a vertical line
        vline!([threshold],subplot=i+1,ls=:dash,color="black")
        annotate_text = L"  \zeta=%$(tolerance[i])"
        top_ = ylims(plot!()[i+1])[2]
        bot_ = ylims(plot!()[i+1])[1]
        lef_ = xlims(plot!()[i+1])[1]
        rih_ = xlims(plot!()[i+1])[2]
        
        y_pos = 
        annotate!(threshold+(rih_-lef_)/20,bot_+(top_-bot_)*0.95,text(annotate_text,6,:left),subplot=i+1)
    end

    h = ylims(plot!())[2]
    plot!(ylims=[0,(1/(1-frac*1.26))*h],subplot=1,xlabel=L"Return Period ( $1/p$ )",ylabel="Return Level")
    
    return plot!(legend=:none)
end

function plot_exceedance_plot(sf;tolerance=1.0)
    return plot_exceedance_plot(plot(),sf;tolerance=tolerance)
end

# +
function my_mrl_plot(mydata;do_linear_fit=false,plot_args=(),do_dots=true,above=1)
    mrlplot_dat = mrlplot_data(mydata)
    
    subset = above:length(mrlplot_dat[:,1])
    
    x_vals = mrlplot_dat[:,1][subset]
    y_vals = mrlplot_dat[:,2][subset]
    y_vals_l = mrlplot_dat[:,3][subset] .- y_vals
    y_vals_u = y_vals .- mrlplot_dat[:,4][subset] 

    if do_dots
        plot!(x_vals,y_vals,ribbon=(y_vals_l,y_vals_u),color=1;plot_args...)
        scatter!(x_vals,y_vals,ribbon=(y_vals_l,y_vals_u),color=1,markersize=2.0, markerstrokewidth=0.0;plot_args...)
    end
    
    if do_linear_fit
        (a,b) = CurveFit.linear_fit(x_vals,y_vals)
        plot!(x_vals,x_vals.*b.+a,color=2;plot_args...) 
    end
    
    f_size = 5
    gf_size = 7
    plot!(;xtickfont = font(f_size),xrotation=25,xguidefontsize=gf_size,
          ytickfont = font(f_size),yguidefontsize=gf_size,plot_args...)
    return plot!()
end

function find_mrl_prop_to_achieve(mydata;tolerance=1.0)
    mrlplot_dat = mrlplot_data(mydata)
    
    mrlplot_dat 
    
    x_vals = mrlplot_dat[:,1]
    y_vals = mrlplot_dat[:,2]
    y_vals_l = mrlplot_dat[:,3]
    y_vals_u = mrlplot_dat[:,4] 
    
    l = length(x_vals)
    
    it_fits = false
    
    if tolerance<0
        return (floor(Int64,(1+tolerance)*length(x_vals)),x_vals)
    end
    
    pos = 0
    while !it_fits
        pos += 1
        
        my_range = pos:l
        
        x_vals_in_range = x_vals[my_range]
        y_vals_in_range = y_vals[my_range]
        y_vals_l_in_range = y_vals_l[my_range]
        y_vals_u_in_range = y_vals_u[my_range]
        
        diff_l = y_vals_in_range .- y_vals_l_in_range
        diff_u = y_vals_u_in_range .- y_vals_in_range
        
        y_vals_l_in_range = y_vals_in_range .- (tolerance .* diff_l)
        y_vals_u_in_range = y_vals_in_range .+ (tolerance .* diff_u)
        
        (a,b) = CurveFit.linear_fit(x_vals_in_range,y_vals_in_range)
        fitted_y_vals = x_vals_in_range.*b.+a
        
        it_fits = all(x->x,y_vals_l_in_range .<= fitted_y_vals .<= y_vals_u_in_range)
        #print(y_vals_l_in_range[end-1]," ",fitted_y_vals[end-1]," ",y_vals_u_in_range[end-1],"\n")
        
    end
    
    return (pos,x_vals)
end

function get_exceedances(mydata,threshold)
    exceedances_pos = ((x) -> x>threshold).(mydata)
    exceedances = mydata[exceedances_pos] .- threshold
    return exceedances
end

function get_first_exceedance_pos(mydata,threshold)
    sorted_data = sort(mydata)
    exceedances_pos = ((x) -> x>threshold).(sorted_data)
    return findfirst(exceedances_pos)
end

function plot_top_prop_mrl(mydata,prop_top;do_linear_fit=false,plot_args=(),do_dots=true)
    l = length(mydata)
    top = floor(Int64, prop_top*l)
    top_data = mydata[top:l]
    return my_mrl_plot(top_data;do_linear_fit=do_linear_fit,plot_args=plot_args,do_dots=do_dots)
end
# -

# # Creating and saving figures

# +
function plot_all_stylized_facts(sf::StylizedFactsPlot; plot_size=(1200, 800), titles_off=false, tolerance=1.0,for_mid=())

    l = @layout [a b c; d e f]

    if titles_off
        my_title = (title="",)
    else
        my_title = ()
    end
    
    (p_mid_price, p_log_returns, p_hist_log_returns,
            p_qq_log_returns, p_indented_acfs, p_exceedance_plot) = 
                generate_stylized_facts(sf; my_title=my_title, tolerance=tolerance,for_mid=for_mid)
    
    
    return plot(p_mid_price,       p_log_returns, p_hist_log_returns, 
                p_qq_log_returns,  p_indented_acfs, p_exceedance_plot,          
                    layout=l, 
                    tickfontsize=6, guidefontsize=8,
                    titlefontsize=10, left_margin=10mm, size=plot_size, dpi=300);
     
end

# -

function save_fig(;folder_name="",file_name="",folder_path="",save_type="svg",
        plot_size=(1200,1200),dpi=300,scale_=0.25,for_plot=(),notify=false)
    if folder_name == "" || file_name == "" || folder_path == ""
        @warn "Need to provide a folder path, folder name and file name"
        return -1
    end
    plot!(size=plot_size,dpi=dpi,scale_=scale_)
    plot!(;for_plot...)
    savefig(string(folder_path,"/",folder_name,"/",file_name,".",save_type))
    
    if notify
        print("Figure saved successfully\n")
    end
end

function save_figs(figs; folder_name="", folder_path="", names, save_type="svg", insert_numbers=false, plot_size=(1200,1200),dpi=300,scale_=0.25, for_plot=(), notify=false)
    
    if insert_numbers #if we should insert an underscore then numbers
        numbers = ((i) -> string("-",i)).(1:length(figs))
    else
        numbers = repeat([""],length(figs)) #blank strings
    end
    
    for i in 1:length(figs)
        plot(figs[i])      
        save_fig(;folder_name=folder_name,file_name=string(names[i],numbers[i]),folder_path=folder_path,save_type=save_type,plot_size=plot_size,dpi=dpi,scale_=scale_,for_plot=for_plot)
    end
    
    if notify
        print("Figures saved successfully\n")
    end
end

function generate_stylized_facts(sf::StylizedFactsPlot; my_title=(), tolerance=1.0,for_mid=())
    
    p_mid_price = plot_mid_price_path(sf;my_title...,for_mid...)
    p_log_returns = plot_log_returns(sf;alpha=0.33,my_title...)
    p_hist_log_returns = plot_hist_log_returns(sf;my_title...)
    
    p_qq_log_returns = plot_qq_log_returns(sf;my_title...)
    p_indented_acfs = plot_indented_acfs(sf)
    p_exceedance_plot = plot_exceedance_plot(sf;tolerance=tolerance)
    
    return [p_mid_price, p_log_returns, p_hist_log_returns,
            p_qq_log_returns, p_indented_acfs, p_exceedance_plot]
end

function save_seperate(sf::StylizedFactsPlot; file_name="", folder_name="", folder_path="", save_type="svg",plot_size=(1200,1200),
        dpi=300,scale_=0.25,for_plot=(),tolerance=1.0,quick=false,for_mid=())
    my_title = (title="",)
    
    (p_mid_price, p_log_returns, p_hist_log_returns,
            p_qq_log_returns, p_indented_acfs, p_exceedance_plot) = 
                generate_stylized_facts(sf; my_title=my_title, tolerance=tolerance,for_mid=for_mid)
    
    #p_mid_price = plot_mid_price_path(sf;my_title...)
    plot(p_mid_price)
    save_fig(;file_name=string(file_name,"_1-mp"),folder_name=folder_name,folder_path=folder_path,             
        save_type=save_type,scale_=scale_,dpi=dpi,plot_size=(plot_size[1]*1.1,plot_size[2]),for_plot=for_plot)
    
    #name = string(folder_path,"/",folder_name,"/",string(file_name,"_1-mp"),".",save_type)
    #img_source = load(name)
    #img_size = size(img_source)
   # 
   # img_cropped = @view img_source[ :,floor(Int, 10/11*img_size[2]) : floor(Int, 7/8*img_size[2])] 
   # save(name,img_cropped)
    
    #p_hist_log_returns = plot_hist_log_returns(sf;my_title...)
    plot(p_hist_log_returns)
    save_fig(file_name=string(file_name,"_3-hlr"), folder_name=folder_name,folder_path=folder_path,  
        save_type=save_type,scale_=scale_,dpi=dpi,plot_size=plot_size,for_plot=for_plot)
    
    
    #p_exceedance_plot = plot_exceedance_plot(sf)
    plot(p_exceedance_plot)
    save_fig(file_name=string(file_name,"_6-expl"),    folder_name=folder_name,folder_path=folder_path,               
        save_type=save_type,scale_=scale_,dpi=dpi,plot_size=plot_size,for_plot=for_plot)
    
    
    
    #p_qq_log_returns = plot_qq_log_returns(sf;my_title...)
    plot(p_qq_log_returns)
    save_fig(file_name=string(file_name,"_4-qqlr"), folder_name=folder_name,folder_path=folder_path,     
        save_type=save_type,scale_=scale_,dpi=dpi,plot_size=plot_size,for_plot=for_plot)
    
    #p_indented_acfs = plot_indented_acfs(sf)
    plot(p_indented_acfs)
    save_fig(file_name=string(file_name,"_5-acfs"), folder_name=folder_name,folder_path=folder_path,               
        save_type=save_type,scale_=scale_,dpi=dpi,plot_size=plot_size,for_plot=for_plot)
    
    #p_log_returns = plot_log_returns(sf;alpha=0.33, my_title...)
    if !quick
        plot(p_log_returns)
        save_fig(;file_name=string(file_name,"_2-lr"), folder_name=folder_name,folder_path=folder_path,          
           save_type=save_type,scale_=scale_,dpi=dpi,plot_size=plot_size,for_plot=for_plot)
    end
    
    
    
    print("Saved figures to ",folder_name," successfully")
end
