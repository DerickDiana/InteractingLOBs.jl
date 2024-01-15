# -*- coding: utf-8 -*-
__precompile__()


module InteractingLOBs

using LinearAlgebra
using Statistics
using Distributions
using Random
using SharedArrays
using Distributed
using SpecialFunctions
using Logging
using IOLogging
using ProgressMeter
#using TSSM
using Roots
using Interpolations
using JLD2
using LsqFit
using LaTeXStrings
using Images
using Parameters

# +
include("MyTypes.jl")
include("../../ContinuousLearning.jl/src/ContinuousLearning.jl")
RLBrain = ContinuousLearning.RLBrain
RLParam = ContinuousLearning.RLParam
RLView = ContinuousLearning.RLView
copy_brain = ContinuousLearning.copy_brain
include("reaction_diffusion_path.jl")
include("data_passer.jl")
include("source_function.jl")
include("coupling_function.jl")
include("rl_push_function.jl")
include("special_function.jl")
include("randomness_function.jl")
include("reaction_diffusion_spde.jl")
include("objective_surface.jl")
include("plot_stylized_facts.jl")
include("useful_functions.jl")


__version__ = "Interacting LOB"
# -

export  SLOB,
        SourceTerm,
        CouplingTerm,
        RLPushTerm,
        RandomnessTerm,
        DataPasser,
        source_function,
        coupling_function,
        rl_push_function,
        randomness_function,
        parse_commandline,
        ObjectiveSurface,
        InteractOrderBooks,
        StylizedFacts,
        to_simulation_time,
        to_real_time,
        clear_double_dict,
        plot_price_impact,
        foo,
        SibuyaKernelModified,
        calculate_modified_sibuya_kernel,
        my_interpolator,
        auto_interpolator,
        get_interpolated_intercept,
        compute_jump_probabilities,









        plot_all_stylized_facts,
        plot_indented_acfs,
        StylizedFactsPlot,
        tick_rule, 
        plot_all_stylized_facts, 
        plot_log_returns,
        plot_hist_log_returns,
        plot_qq_log_returns, 
        plot_acf_order_flow, 
        plot_acf_log_returns,
        plot_acf_abs_log_returns,
        plot_indented_acfs,
        plot_exceedance_plot,

        gp_qq_plot,
        gp_density_plot,
        gp_probability_plot,
        gp_return_level_plot,

        my_mrl_plot,
        find_mrl_prop_to_achieve,
        get_exceedances,
        get_first_exceedance_pos,
        plot_top_prop_mrl,

        save_seperate,
        save_fig, save_figs,











        generate_Δts_exp,
        calculate_Δt_from_Δx,
        calculate_Δx_from_Δt,


        change_side,



        get_sums,
        plot_sums,
        get_second_derivative,
        myvar,
        mymean,
        plot_price_path,
        plot_density_visual,
        quick_plot,
        show_path,
        calculate_price_impacts,
        fit_and_plot_price_impact,
        get_length_of_time_that_allows_kick,
        difference_checker,
        save_data,
        load_data,
        obtain_data_list,
        obtain_price_impacts,
        obtain_seed,

        my_repeat,

        calculate_area,
        calculate_distance_to_get_volume,
        calculate_intercept,
        calculate_trapezium_area,
        calculate_trapezium_area_many,

        my_pad, get_layout_for_n_plots,

        price_to_index, index_to_price,

        derivative_at, get_central_derivative, get_effective_market_orders,

        #fit_and_plot_price_impact_mod,
        #obtain_price_impacts_mod,
        #obtain_price_impacts_mod2,
        #calculate_price_impacts_mod,
        #calculate_price_impacts_mod2,
        #fit_and_plot_price_change_mod
        obtain_price_impacts,
        calculate_price_impacts,
        fit_and_plot_price_impact,
        fit_and_plot_price_change,

        my_power_fit, my_log_fit, f_unc, mr,
        arrow_from_abs_to_frac, arrow_from_abs_to_frac,
        draw_square_abs,

        RLParam,RLBrain

end # module
