# -*- coding: utf-8 -*-
using SequentialLOB

# +
function main(output = "stdout")
    parsed_args = parse_commandline()
    slob_model = SLOB(
        1,
        parsed_args["T"],
        parsed_args["p₀"],
        parsed_args["M"],
        parsed_args["L"],
        parsed_args["D"],
        parsed_args["σ"],
        parsed_args["nu"],
        parsed_args["α"],
        SourceTerm(parsed_args["λ"], parsed_args["μ"]),
    )
    if output == "stdout"
        mid_price_paths = slob_model(parsed_args["SEED"])
        print(mid_price_paths[:, 1])
    else
        return slob_model(parsed_args["SEED"])
    end
    
end
# -


if "" != PROGRAM_FILE && occursin(PROGRAM_FILE, @__FILE__)
    main()
end
