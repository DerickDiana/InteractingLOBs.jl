# -*- coding: utf-8 -*-
using Main
Main.include("../../ContinuousLearning.jl/src/ContinuousLearning.jl")
cntl = Main.ContinuousLearning
using Plots
using SimpleDirectMediaLayer
using SimpleDirectMediaLayer.LibSDL2



# +
#import Pkg;
#Pkg.add(Plot)
# -

using Parameters
simple_pendulum = true #are we simulating a pendulum swing back and forth, or a pendulum on a cart?
method = 1 
# method = 1 represents using Actor Critic
# method = 2 represents using known Input Gain Matrix
# method = 3 represents using unknown (numeric) Input Gain Matrix

# # Physics

# +
# physics parameters
# https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html

x = 0.0
∂x = 0.0
∂∂x = 0.0
θ_0 = 1.0*pi
θ = θ_0
∂θ = 0.0
∂∂θ = 0.0
f_ = 0.0
f_c = 0

if simple_pendulum
    set = (l=1.0,μ_c=0.3,μ_p=0.01,m=1.0,M=1.5,g=9.8)  
else
    set = (l=1.0,μ_c=0.3,μ_p=0.01,m=1.0,M=1.5,g=9.8)  
end
#μ_p is friction between the pole and the point its attached to
#μ_c is the friction between the cart and road
#l is the length of the pole
#m is the mass of the pole while M is the mass of the cart
#g is g

dt = 0.02
sc = 100 # scaling factor that determines how things render
t = 0.0
p_t = 0.0
#####################################################################################
if simple_pendulum
    ∂f_u = (x_s) -> [0,1/(set.m*set.l^2)]
else
    ∂f_u = (x_s) -> [0,-(7/3)/(set.m * cos(cntl.mod_(x_s[3],-pi,-pi,2*pi))^2 - 7/3*set.M),0,0]
end
# -

if simple_pendulum
    x_s = vcat(0,0)
else
    x_s = vcat(0,0,0,0)
end

# +
function runge_kutta(f,t,x,h;a,b,c)
    @assert c[1]==0
    
    for i in 1:length(c)
        k[i] = f(t+c[i],x +   sum(  ((j)->(   a[i,j] * k[j]   )).(i-1:(-1):1) )*h   )
    end

end
# -

function ∂∂θ_(x,∂x,∂∂x,θ,∂θ,∂∂θ,f;l,μ_c,μ_p,m,M,g)
    return 3/(7*l) * (g * sin(θ) - ∂∂x * cos(θ) - μ_p * ∂θ / (m * l))
end

function ∂∂x_(x,∂x,∂∂x,θ,∂θ,∂∂θ,f;l,μ_c,μ_p,m,M,g)
    num = m*g*sin(θ)*cos(θ) - 7/3*(f + m * l * ∂θ^2 * sin(θ) - μ_c * ∂x) - μ_p * ∂θ * cos(θ) / l
    denom = m * cos(θ)^2 - 7/3 * M
    
    return num/denom
end

function ∂∂θ_p(x,∂x,∂∂x,θ,∂θ,∂∂θ,f;l,μ_c,μ_p,m,M,g)
    return 1/(m*l^2) * (  -μ_p*∂θ + m * g * l * sin(θ) + f    )
end

function ∂∂x_p(x,∂x,∂∂x,θ,∂θ,∂∂θ,f;l,μ_c,μ_p,m,M,g)
    return 0
end
n_kick = 0.0

# +
function updatephysics(x,∂x,∂∂x,θ,∂θ,∂∂θ,f,set,dt,f_,t,g;do_presses=false,do_random=true)
    global n_kick, simple_pendulum
    # compute the force
    f_ = 0  
    if do_random
        #f_ = (rand()-0.5) * 10.0 #allow for random forces
        
        n_kick = λ*n_kick + rand(Normal(0.0,1.0))*(1-λ)
        f_ = n_kick * 10.0
    end
    
    if do_presses
        if g.keyboard.V
            amount = 2.0
        else
            if g.keyboard.C
                amount = 10.0
            else 
                amount = 5.0
            end
        end

        if g.keyboard.J
            f_ += -amount
        end

        if g.keyboard.K
            f_ += +amount
        end
    end
    
    # step the system
    
    #####################################################################################
    if simple_pendulum
        # pendulum physics (p for pole/pendulum)
        ∂∂x = ∂∂x_p(x,∂x,∂∂x,θ,∂θ,∂∂θ,f+f_;set...)
        ∂∂θ = ∂∂θ_p(x,∂x,∂∂x,θ,∂θ,∂∂θ,f+f_;set...)
    else
        # cart pole physics
        ∂∂x = ∂∂x_(x,∂x,∂∂x,θ,∂θ,∂∂θ,f+f_;set...)
        ∂∂θ = ∂∂θ_(x,∂x,∂∂x,θ,∂θ,∂∂θ,f+f_;set...)
    end
    
    ∂x = ∂x + dt * ∂∂x
    ∂θ = ∂θ + dt * ∂∂θ
    
    x = x + dt * ∂x
    θ = θ + dt * ∂θ
    if θ==NaN
        print("Disaster")
    end
    
    
#     if x > WIDTH/2/sc
#         x = -WIDTH/2/sc
#     end
    
#     if x < -WIDTH/2/sc
#         x = WIDTH/2/sc
#     end
    
    return x,∂x,∂∂x,θ,∂θ,∂∂θ,f,set,dt,f_,t
end
# -

# # Visuals

# +
mutable struct my_text_actor 
    font_name
    font_size
    font_color
    font
    actor
    pos
    text
end

function my_text_actor(text,pos,font_name,font_size,font_color)
font = TTF_OpenFont(pwd()*"/fonts/"*font_name, font_size)
    actor = TextActor(text,font_name;color=font_color)
    
    return  my_text_actor(font_name,font_size,font_color,font,actor,pos,text)
end

function modify_text_of_actor(text_actor,text)
    sf = TTF_RenderText_Blended(text_actor.font, text , SDL_Color(text_actor.font_color...))
    w,h = size(sf)
    text_actor.actor.surface = sf
    text_actor.actor.position = Rect(text_actor.pos[1],text_actor.pos[2],Int(w),Int(h))
end

function draw(text_actor::my_text_actor)
    draw(text_actor.actor)
end

# +
mutable struct my_image_actor
    width
    height
    pos
    dir_string
    image_name
    actor
end

function my_image_actor(image_name,size_,pos,scale_factor,dir_string)
    plot([0,1])
    width = size_[1]
    height = size_[2]
    plot!(size=(width,height))
    savefig(dir_string*image_name)
    
    actor = Actor(image_name,pos=pos,scale = [scale_factor,scale_factor])

    return my_image_actor(width,height,pos,dir_string,image_name,actor)
end

function draw_plot_to_image(image_actor)
    plot!(size=(image_actor.width,image_actor.height))
    path = image_actor.dir_string*image_actor.image_name
    savefig(path)
    image_actor.actor.surface = IMG_Load(path) 
    #here we can directly put the image we want in
end

function draw(image_actor::my_image_actor)
    draw(image_actor.actor)
end

# +
### Set canvas size:
WIDTH = 1200
HEIGHT = 800

### Set up custom objects to show plots:
# needs these constants
dir_string = "/home/../tmp/"

# and here are the objects:
image1 = my_image_actor("img_for_plot.png",(600,400),(WIDTH - 0.5*600,0),0.5,dir_string)

# which will plot this data:
all_ys_so_far = zeros(0)

### Set up custom objects to show text:
# needs these constants
colors = range(colorant"black", colorant"white")

font_name = "ubuntuth.ttf"
font_size = 24
font_color = Int[0,0,0,255]

# and will plot this often:
plot_every = 1000
last_plot = 0

# and here are the objects
displaytime1 = my_text_actor("0",(WIDTH/20,HEIGHT/20),font_name,font_size,font_color)
displaytime2 = my_text_actor("0",(WIDTH/20,2.5*HEIGHT/20),font_name,font_size,font_color)

txtact_fast = TextActor(">>>>",font_name;color=font_color)
txtact_fast.pos = (WIDTH/20,3.5*HEIGHT/20)
txtact_play = TextActor(">",font_name;color=font_color)
txtact_play.pos = txtact_fast.pos


# +
function draw(g::Game)
    global x,θ,f,set,f_,t,p_t,txtact_fast,txtact_play,t_arr,t_arr2,cum_r,mod,
    δ,slow_end,prevtemp,prevtemp2,show,x_s,r,episode_length,
    #w_A,n_A,w_V,r,c,s,V,Vmax,Vmin,
    #∂b_x,b,
    RLBrain_,RLParam_,
    image1,all_ys_so_far,last_plot,plot_every,
    font,font_color
    
    @unpack w_A,w_V,w_M,e,A,V,σ,n_A = RLBrain_;
    @unpack c,s,η_A,η_V,Vmax,Vmin,γ,λ,τ,σ_0,dt,init_brain,f_0,f_total = RLParam_;
    
    clear()
    
    ### Computations for drawing plots:

    if t-last_plot > plot_every
        plot(all_ys_so_far)
        
        draw_plot_to_image(image1)
        
        last_plot = t
    end
    
    draw(image1)
    
    ### Computations for drawing text:
    
    modify_text_of_actor(displaytime1,  "Time: "*chop(  string(round(t,digits=0)) , tail = 2)  )
    draw(displaytime1)
    
    modify_text_of_actor(displaytime2,  "Trial: "* chop(  string(round(t/episode_length,digits=0)) , tail = 2)  )
    draw(displaytime2)
    
    if (t <= slow_end)||show
        draw(txtact_play)
    else
        draw(txtact_fast)
        return
    end
    
    ### Computations for drawing the cart and what not:
    cart_width = 40
    cart_height = 10
    l = set[1]
    c_pos = WIDTH/2 + floor(Int64,sc*x)
    pole_dis_x = floor(Int64,sc*l*cos(-θ+pi/2))
    pole_dis_y = floor(Int64,sc*l*sin(-θ+pi/2))
    
    
    draw(Line(-WIDTH/2,HEIGHT/2,WIDTH,HEIGHT/2),colorant"grey")
    
    function draw_pole_at(pos)
        draw(Rect(pos-cart_width/2,HEIGHT/2-cart_height/2,cart_width,cart_height),fill=true)
        draw(Line(pos,HEIGHT/2,pos+pole_dis_x,HEIGHT/2-pole_dis_y))
        draw(Circle(pos+pole_dis_x,HEIGHT/2-pole_dis_y,3),fill=true)
        draw(Circle(pos,HEIGHT/2,sc*l),colorant"grey")
    end
    
    draw_pole_at(c_pos)
#     draw_pole_at(c_pos+WIDTH)
#     draw_pole_at(c_pos-WIDTH)
    
    drawfire(c_pos,HEIGHT/2,f_,cart_width;color=colorant"purple")
    drawfire(c_pos,HEIGHT/2,f,cart_width;color=colorant"orange")
    
#     if length(size(shape_b))>2
#         drawfire(c_pos-WIDTH,HEIGHT/2,f_,cart_width;color=colorant"purple")
#         drawfire(c_pos-WIDTH,HEIGHT/2,f,cart_width;color=colorant"orange")

#         drawfire(c_pos+WIDTH,HEIGHT/2,f_,cart_width;color=colorant"purple")
#         drawfire(c_pos+WIDTH,HEIGHT/2,f,cart_width;color=colorant"orange")
#     end
    
    
    
    dis_h = 0.15*HEIGHT
    dis_h_pos = HEIGHT-1.4*dis_h
    dis_h_pos2 = 0 
    dis_w_pos = WIDTH/2-dis_h/2
    
    if method == 3
        (b,∂b_x,∂b_u) = cntl.b____(x_s,f;c,s) #how far are we from each basis vector
    else
        (b,∂b_x) = cntl.b____(x_s;c,s) #how far are we from each basis vector
    end
    
    shape_b = reshape(b,size(c)...)

    # Draw bottom left block
    temp = slice_dir(shape_b,1,2)
    maxpos = findmax(temp)[1] 
    if maxpos != 0
        temp = temp./maxpos
        temp = (1).-temp
    end

    drawgrid(temp,dis_w_pos-2*dis_h,dis_h_pos,dis_h,dis_h)


    # Draw top left block
    if length(size(shape_b))>2 && !(method==3) 
                #size(shape_b)>2 indicates that we are using the cart as well which has
                #4 co-ordinates. But this is only true if we didn't already put a third 
                #co-oridinate by using method 3 for u
        temp = slice_dir(shape_b,3,4)
        maxpos = findmax(temp)[1] 
        if maxpos != 0
            temp = temp./maxpos
            temp = (1).-temp
        end

        drawgrid(temp,dis_w_pos-2*dis_h,dis_h_pos2,dis_h,dis_h)
    end

# draw weights of value function
#     temp = reshape(w_V,size(c)...)
#     maxpos = findmax(temp)[1]
#     minpos = findmin(temp)[1]
#     if maxpos-minpos != 0
#         temp = (temp.-minpos)./(maxpos-minpos)
#     end
    
#     drawgrid(temp,dis_w_pos,dis_h_pos,dis_h,dis_h)
    
    # draw weights of action function
    # temp = reshape(w_A,size(c)...)
    # maxpos = findmax(temp)[1]
    # minpos = findmin(temp)[1]
    # if maxpos-minpos != 0
    #     temp = (temp.-minpos)./(maxpos-minpos)
    # end
   
    # drawgrid(temp,dis_w_pos+2*dis_h,dis_h_pos,dis_h,dis_h)
    
    t_test =  t
    t_test = t_test%0.5
    t_test = t_test - floor(Int64,t_test)
    
    
    tick = ((t<=slow_end) && (t_test<=dt))||show
    tempval = 0
    tempderiv = 0
    
    # if tick 
    #     res = ((z) -> b___(z;c,s)).(c)
    #     res = slice_dir(res,3,4)
    
    # end
    
    res = 0
    if tick 
        res_w_V = reshape(w_V,size(c)...)
        res_w_A = reshape(w_A,size(c)...)
        #     tempval = ((z) -> sum(w_V.*z[1])).(res)
        #     tempderiv = ((z) -> sum(w_V.*z[2])[2]).(res)
    end
    
    # Draw top middle block
    if length(size(shape_b))>2 && !(method==3)
        if tick
            temp = slice_dir(res_w_V,3,4)
            maxpos = findmax(temp)[1]
            minpos = findmin(temp)[1]
            if maxpos-minpos != 0
                temp = (temp.-minpos)./(maxpos-minpos)
            end
            prevtemp = temp
        end
        drawgrid(prevtemp,dis_w_pos,dis_h_pos2,dis_h,dis_h)
    end
    
            
    # Draw bottom middle block
    if tick
        temp = slice_dir(res_w_V,1,2)
        maxpos = findmax(temp)[1]
        minpos = findmin(temp)[1]
        if maxpos-minpos != 0
            temp = (temp.-minpos)./(maxpos-minpos)
        end
        prevtemp = temp
    end
    drawgrid(prevtemp,dis_w_pos,dis_h_pos,dis_h,dis_h)
    
    # Draw bottom right block
    if tick
        temp = slice_dir(res_w_A,1,2)
        maxpos = findmax(temp)[1]
        minpos = findmin(temp)[1]
        if maxpos-minpos != 0
            temp = (temp.-minpos)./(maxpos-minpos)
        end
        prevtemp2 = temp
    end
    drawgrid(prevtemp2,dis_w_pos+2*dis_h,dis_h_pos,dis_h,dis_h)
    
    
    
    
    
#     if length(size(shape_b))<=2
#         #draw derivative of value function in direction ...
#         if tick
#             #temp = ((z) -> sum(w_V.*b___(z;c,s)[2])[2]).(c)
#             maxpos = findmax(tempderiv)[1]
#             minpos = findmin(tempderiv)[1]
#             if maxpos-minpos != 0
#                 tempderiv = (tempderiv.-minpos)./(maxpos-minpos)
#             end
#             prevtemp2 = tempderiv
#         end
#         drawgrid(prevtemp2,dis_w_pos+2*dis_h,dis_h_pos,dis_h,dis_h)
#     end
 
#     print("here")
    
end
# -

using Distributions

# learning parameters



# +
function drawfire(x1,y1,mag,width;scale = 2.0,color=colorant"orange")
    depth = floor(Int64,mag*scale)
    height = 10
    
    if mag==0
        return 
    end
    
    if mag>0
        rect_shift = 1 #need to shift rectangle cause of how co-ordinates work
        which_side = 1 #which side should the fire go
    else
        rect_shift = 0
        which_side = -1
        depth=-depth
    end
    
    for (t,d) in [(1.0,1.0),(0.7,1.5),(0.5,2.0)]
    
        depth_  = floor(Int64,depth*d)
        height_ = floor(Int64,height*t)
        x_pos   = floor(Int64,x1-which_side*width/2-depth_*rect_shift)
        y_pos   = floor(Int64,y1-height_/2)
   
        draw(Rect(x_pos,y_pos,depth_,height_),color,fill=true) 
    end
    
end

# +
# f_arr = () -> [
#     TextActor("0","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("1","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("2","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("3","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("4","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("5","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("6","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("7","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("8","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("9","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor(".","ubuntuth.ttf";color=Int[0,0,0,255]),
#     TextActor("-","ubuntuth.ttf";color=Int[0,0,0,255])
# ]

# t_arr = repeat([f_arr()],20)
# t_arr2 = repeat([f_arr()],20)

# +
# function drawnumber(t,WIDTH,HEIGHT,t_arr;digits=1)
#     t_ = round(t,digits=digits)
#     t_s = string(t_)
#     a = length(t_s)
    
#     i = 1
#     for s in reverse(t_s)
#         if s=='.'
#             temp = t_arr[i][11]
#         elseif s=='-'
#             temp = t_arr[i][12]
#         else
#             num = parse(Int64,s)
#             temp = t_arr[i][num+1]
#         end
        
#         temp.pos = (WIDTH+(a-i)*12,HEIGHT)
#         draw(temp)
#         i += 1
        
#     end
# end
# -

function drawgrid(values,xpos,ypos,height,width;do_block=true)
    
    mf = (v) -> floor(Int64,v)
    
    
    xlen = size(values)[1]
    ylen = size(values)[2]
    bwidth = mf(width/xlen)
    bheight = mf(height/ylen)
    xpos = mf(xpos)
    ypos = mf(ypos)
    onecol = colorant"black"
    zerocol = colorant"grey"
    
    for x_ in 1:xlen
        for y_ in 1:ylen
            currcol = colors[mf(98*values[x_,y_])+2]
            draw(Rect(xpos+(bwidth+1)*x_,ypos+(bheight+1)*y_,bwidth,bheight),currcol,fill=true)
        end
    end
    
    if do_block
        draw(Rect(xpos+bwidth+1,ypos+bheight+1,(bwidth+1)*xlen,(bheight+1)*ylen))
    end
end

# # Update function

# +
function update(g::Game)
    global x,∂x,∂∂x,θ,∂θ,∂∂θ,set,t,dt,f,f_,era_count,era_length,episode_count,episode_length,
    quick_end,slow_end,episode_end,punish_length,punish_end,slow_episode_num,
    punish,pun_r,cum_r,x_s,
    mod,b,∂b_x,show,RLBrain_,RLParam_,
    A,w_A,V,w_V,
    r,
    n_A,τ,σ,c,s,η_A,η_V,e,σ_0,Vmax,Vmin,γ,λ,dt,init_brain,f_0,f_total,f,
    simple_pendulum, method, Vmax, Vmin,
    adjusted_max_reward_per_episode, adjusted_min_reward_per_episode,
    all_ys_so_far
    
    while t < quick_end
        
        show = false
        do_presses = t < slow_end
        
        if t >= episode_end
            t = episode_end
            
            x,∂x,∂∂x,θ,∂θ,∂∂θ,f = (0,0,0,θ_0+0*(rand()-0.5)*pi,0,0,0)
            
            episode_end = episode_count*episode_length + episode_length
            episode_count += 1
            
            punish = false
            
            mod = Main.ContinuousLearning.bound_output(
                (adjusted_max_reward_per_episode - cum_r)/(adjusted_max_reward_per_episode-adjusted_min_reward_per_episode),
                0,1)
            #print(mod,"\n")
            cum_r = 0
        end
        
        if !punish
            if abs(θ-pi)>4*pi || (abs(x) > WIDTH/2/sc)
                punish = true
                punish_end = t + punish_length
            end
        end
        
        if !punish
            x,∂x,∂∂x,θ,∂θ,∂∂θ,f,set,dt,f_,t = updatephysics(x,∂x,∂∂x,θ,∂θ,∂∂θ,f,set,dt,f_,t,g;do_presses=do_presses,do_random=false)
            r = cos(cntl.mod_(θ,-pi,-pi,2*pi))
            append!(all_ys_so_far,r)
        else
            r = pun_r
        end
        prev_x_s = x_s[:]
        
        if simple_pendulum
            x_s = vcat(cntl.mod_(θ,-pi,-pi,2*pi),∂θ)
        else
            x_s = vcat(x,∂x,cntl.mod_(θ,-pi,-pi,2*pi),∂θ)
        end
        
        
        # Store the information about the environment, current reward and current action in the correct format
        RLView_ = cntl.RLView(x_s,r,f)
        
        
        # learning happens
        if method == 1
            f,RLBrain_ = cntl.updatelearningAC(RLView_,RLBrain_,RLParam_)
        elseif method == 2
            f,RLBrain_ = cntl.updatelearningACknownIG(RLView_,RLBrain_,RLParam_,∂f_u,mod)
        elseif method == 3
            x_dot = (x_s - prev_x_s)/dt
            f,RLBrain_ = cntl.updatelearningAClearnedIG(RLView_,RLBrain_,RLParam_,x_dot)
        end
        #f = 0
        
        if punish && (t >= punish_end)
            t = episode_end
            continue
        end
        
        cum_r += r
        t += dt
        
        if do_presses
            return
        end
        
        temp_t = t + 5
        int_t = floor(Int64,temp_t)
        if (int_t%10==0) && (temp_t-int_t<=dt)
            show = true
            return
        end
        
    end
    
    t = quick_end
    era_count += 1
    quick_end = era_count*era_length + era_length
    slow_end =  era_count*era_length + slow_episode_num * episode_length
    
end

#####################################################################################
#f,A,w_A,e,V,w_V,n_A,b,Vmax,Vmin = updatelearningAC(vcat(x,∂x,θ,∂θ),r,(w_A,A,η_A),(w_V,V,e,η_V),c,s,dt,τ,σ_0,n_A,Vmax,Vmin,γ,λ   )
#f,A,w_A,e,V,w_V,n_A,b,Vmax,Vmin = cntl.updatelearningAC(vcat((cntl.mod_(θ,-pi,-pi,2*pi)),∂θ),r,(w_A,A,η_A),(w_V,V,e,η_V),c,s,dt,τ,σ_0,n_A,Vmax,Vmin,γ,λ   )
#f,e,V,w_V,n_A,b,Vmax,Vmin,∂b_x = updatelearningACknownIG(vcat((mod_(θ,-pi,-pi,2*pi)),∂θ),r,(w_A,A,η_A),(w_V,V,e,η_V),c,s,dt,τ,σ_0,n_A,Vmax,Vmin,γ,λ   )
#f,e,V,w_V,n_A,b,Vmax,Vmin,∂b_x = updatelearningACknownIG(vcat(x,∂x,mod_(θ,-pi,-pi,2*pi),∂θ),r,(w_A,A,η_A),(w_V,V,e,η_V),c,s,dt,τ,σ_0,n_A,Vmax,Vmin,γ,λ   )
#####################################################################################
# -

# # Learning

# +
# program parameter
episode_length = 20
episode_count = 0
episode_end = episode_count*episode_length+episode_length
slow_episode_num = 1

era_length = 2000#500#2000
era_count = 0
quick_end =  era_count*era_length + era_length #0
slow_end  =  era_count*era_length #+ slow_episode_num * episode_length

pun_r = -3.0
punish_length = 3
punish = false
punish_end = 0

# +
len = 15 # for θ and ∂θ
len2 = 8 # for x and ∂x
len3 = 2 # for u

θ_range = range(-pi,pi,length=len)
#∂θ_range = range(-3*pi,3*pi,length=len)
∂θ_range = range(-2.5*pi,2.5*pi,length=len) # they say to use this range for ∂θ
if !simple_pendulum
    x_range = range(-WIDTH/sc/1.5,WIDTH/sc/1.5,length=len2)
    ∂x_range = range(x_range[1]/1.5,x_range[end]/1.5,length=len2)
end

if method == 3
    u_range = range(-2,2,length=len3)
end

#####################################################################################
if simple_pendulum
    if method == 3 
        c = collect(Iterators.product(θ_range,∂θ_range,u_range));
    else
        c = collect(Iterators.product(θ_range,∂θ_range));
    end
else
    if method == 3
        c = collect(Iterators.product(x_range,∂x_range,θ_range,∂θ_range,u_range));
    else
        c = collect(Iterators.product(x_range,∂x_range,θ_range,∂θ_range));
    end
end
c = ((x)->collect(x)).(c)

# +
#tempf = (f) -> (f*1.5)^(-1.0)
tempf = (f) -> (f*1.1)^(-1.0)
tempf2 = (f) -> (f*1.0)^(-1.0)

θ_range_s  =  repeat([tempf(  (θ_range[end]-θ_range[1])/len)],len)
∂θ_range_s =  repeat([tempf(  (∂θ_range[end]-∂θ_range[1])/len)],len )

if !simple_pendulum
    x_range_s  =  repeat([tempf2(  (x_range[end]-x_range[1])/len2)],len2)
    ∂x_range_s =  repeat([tempf2(  (∂x_range[end]-∂x_range[1])/len2)],len2 )
end

if method == 3
    u_range_s = repeat([tempf(  (u_range[end]-u_range[1])/len3)],len3 )
end

#####################################################################################
if simple_pendulum
    if method == 3
        s = collect(Iterators.product(θ_range_s,∂θ_range_s,u_range_s));
    else
        s = collect(Iterators.product(θ_range_s,∂θ_range_s));
    end
else
    if method == 3
        s = collect(Iterators.product(x_range_s,∂x_range_s,θ_range_s,∂θ_range_s,u_range_s));
    else
        s = collect(Iterators.product(x_range_s,∂x_range_s,θ_range_s,∂θ_range_s));
    end
end
s = ((x)->collect(x)).(s)
# -

function slice_dir(mat,dim1,dim2;defaults=-1)
    
    mysize = size(mat)
    no_dim = length(mysize)
    @assert 0 < dim1 <= no_dim
    @assert 0 < dim2 <= no_dim
    @assert dim1 != dim
    
    
    if defaults == -1
        defaults = ((i) -> [ceil(Int64,mysize[i]/2)]).(1:no_dim)
    else
        defaults = ((x) -> [x]).(defaults)
    end
    
    my_slicer = defaults
    
    my_slicer[dim1] = 1:mysize[dim1]
    my_slicer[dim2] = 1:mysize[dim2]
    
    reshape(mat[my_slicer...],mysize[dim1],mysize[dim2])
end

# +
#mat = reshape(1:50,5,5,2)

# +
#slice_dir(mat,2,3)

# +
# Actor
initial_w_scale = 0.0
w_A = (rand(length(c)).*initial_w_scale*2).-initial_w_scale
A = 0.0
#η_A = 5.0
η_A = 0.1
n_A = 0.0
b = (w_A[:].*0)
∂b_x = repeat([[0,0]],length(w_A))

# Critic
initial_w_scale = 0.0
w_V = (rand(length(c)).*initial_w_scale*2).-initial_w_scale
V = 0.0
Vmax = +1.0
Vmin = -0.0
e = repeat([0.0],length(w_V))
η_V = 1.0 

r = 0.0
cum_r = -1.0 * episode_length/dt
σ_0 = 0.5
mod = Main.ContinuousLearning.bound_output((Vmax-V)/(Vmax-Vmin),0,1)
σ = σ_0 * mod

δ = 0
f = 0.0

w_M = [Vector{Float64}(undef,2) for _ in 1:2]
RLBrain_ = cntl.RLBrain(w_A,w_V,w_M,e,A,V,σ,n_A);
f_0 = 5
f_total = 1

# +
dt = 0.02
τ = 1.0
κ = τ/10

if τ==dt
    @assert κ == τ 
    
    γ = 0.0
else
    γ=(1-dt/κ)/(1-dt/τ)
end
λ=(1-dt/τ);
#(γ,λ) = (λ,γ)
(γ,λ)
# -

steps_per_episode = episode_length / dt
absolute_max_reward_per_episode = steps_per_episode * Vmax #reward if it spent all its time in the best state
absolute_min_reward_per_episode = steps_per_episode * Vmin #reward if it spent all its time in the worst state
adjusted_max_reward_per_episode = absolute_max_reward_per_episode / 2
adjusted_min_reward_per_episode = absolute_min_reward_per_episode / 2

RLParam_ = cntl.RLParam(c,s,η_A,η_V,Vmax,Vmin,γ,λ,τ,σ_0,dt,
                        RLBrain_,f_0,f_total);

# +
# using Parameters

# @with_kw mutable struct RLBrain
#     w_A::Vector{Float64}=Vector{Float64}(undef,0)
#     w_V::Vector{Float64}=Vector{Float64}(undef,0)
#     A::Float64=0.0
#     V::Float64=0.0
#     r::Float64=0.0
#     σ::Float64=0.0
#     e::Float64=0.0
#     n_A::Float64=0.0
# end

# RLBrain(w_A,w_V,
#         0.0,0.0,0.0,0.0,0.0,0.0)
# -

# # Junk

function b_(x_s;c,s)
    a = (
        (j) -> exp(-(   mynorm(s[j].*(  arrmod(x_s.-c[j])  )   ) ) )
        ).(1:length(s))
    a / sum(a)
end

function b__(x_s;c,s)
    l = length(s)
    m = length(x_s)
    inner = (
        (k) -> mynorm(s[k].*(  arrmod(x_s.-c[k])  )   )
        ).(1:l)                           #length l, entry k
    a = exp.((0).-inner)                  #length l, entry k
    ∂a_x_inner = ((-2)*a).*(sqrt.(inner)) #length l, entry k
    ∂a_x = repeat([s[1]],l) .* ∂a_x_inner #length l, entry k, then 2x1, entry i
    
    sum_a = sum(a)        #length 1
    sum_∂a_x = sum(∂a_x)  #length 2, entry i
    
    b = a / sum(a)        #length l, entry k
    ∂b_x = ((k) -> (    (∂a_x[k] * sum_a) .+ (sum_∂a_x * a[k]))    ).(1:l)
    ∂b_x = ∂b_x/(sum_a)^2  #length l, entry k, then 2x1, entry i
    return (b,∂b_x)
end

function b___(x_s;c,s)
    l = length(s)
    m = length(x_s)
    inner = (
        (k) -> mynorm(s[k].*(  arrmod(x_s.-c[k])  )   )
        ).(1:l)                           #length l, entry k
    
    
    a = exp.((0).-inner)                  #length l, entry k
    sum_a = sum(a)        #length 1
    
    
    ∂a_x_inner = ((-2)*a)                 #length l, entry k
    ∂a_x = repeat([s[1]],l) .* ∂a_x_inner #length l, entry k, then 2x1, entry i
    sum_∂a_x = sum(∂a_x)  #length 2, entry i
    
    b = a / sum(a)        #length l, entry k
    ∂b_x = ((k) -> (    (∂a_x[k] * sum_a) .- (sum_∂a_x * a[k]))    ).(1:l)
    ∂b_x = ∂b_x/(sum_a)^2  #length l, entry k, then 2x1, entry i
    return (b,∂b_x)
end
