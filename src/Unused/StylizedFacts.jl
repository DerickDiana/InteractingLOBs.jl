module StylizedFacts

using Plots
using Statistics
using Distributions
using StatsPlots
using StatsBase
using Plots.PlotMeasures


include("order_classification.jl")
include("plot_stylized_facts.jl")

export  StylizedFactsPlot, 
        tick_rule, 
        plot_all_stylized_facts, 
        hist_log_returns,
        plot_qq_log_returns, 
        plot_acf_order_flow, 
        plot_acf_log_returns,
        plot_acf_abs_log_returns

end # module
