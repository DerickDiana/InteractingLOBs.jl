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
            #previous_tick_change = 1
            previous_tick_change = -1
        end
    end
    return signs
end
