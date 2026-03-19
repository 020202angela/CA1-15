############################################################
# MASTER TRAINING (HIGH-PRECISION UDE)
############################################################

using CSV, DataFrames, Random
using DifferentialEquations, SciMLSensitivity
using Lux, ComponentArrays
using OptimizationOptimisers
using Statistics, Interpolations, Serialization
using Zygote
import Optimisers

Random.seed!(42)

const M = 1.5f0
const g = 9.81f0
const c_drag = 0.8f0
const BATCH_SIZE = 4
const EPOCHS = 120

softplus(x) = log1p(exp(clamp(x, -20f0, 20f0)))

############################################################
# DATA LOADING
############################################################

function load_dataset(dir)
    files = sort(filter(f -> endswith(f,".csv"), readdir(dir, join=true)))
    data_out = []
    for f in files
        df = CSV.read(f, DataFrame)
        t_vals = Float32.(df.time)
        push!(data_out, (
            t = t_vals,
            h = Float32.(df.altitude),
            v = Float32.(df.velocity),
            itp = LinearInterpolation(t_vals, Float32.(df.thrust), extrapolation_bc=Interpolations.Flat())
        ))
    end
    return data_out
end

data = load_dataset("data")

############################################################
# MODELS
############################################################

nn = Lux.Chain(Lux.Dense(2=>64, tanh), Lux.Dense(64=>64, tanh), Lux.Dense(64=>1))
nn_ps, nn_st = Lux.setup(Random.default_rng(42), nn)

function ground_effect(h, v, p, st)
    scale = softplus(p.log_scale)
    decay = softplus(p.log_decay)
    beta  = softplus(p.log_beta)
    gate  = 1f0/(1f0+exp(clamp((h-0.5f0)/0.1f0, -50f0,50f0)))
    f_phys = scale*exp(-decay*clamp(h,0f0,10f0))*(1 + beta*clamp(v,-10f0,10f0)^2)
    out, _ = nn(vcat(clamp(h,0f0,10f0), clamp(v,-10f0,10f0)), p.nn, st)
    return clamp(gate*(f_phys + 10f0*tanh(out[1])), 0f0, 60f0)
end

function simulate(traj, p_in; mode="ude")
    function rhs!(du, u, p, t)
        T_val = traj.itp(t)
        F = if mode=="baseline"
            0f0
        elseif mode=="blackbox"
            res,_ = nn(vcat(u[1],u[2]), p.nn, nn_st)
            10f0*tanh(res[1])
        else
            ground_effect(u[1], u[2], p, nn_st)
        end
        du[1] = -u[2]
        du[2] = g - (T_val + c_drag*u[2] + F)/M
    end
    prob = ODEProblem(rhs!, [traj.h[1], traj.v[1]], (traj.t[1], traj.t[end]), p_in)
    solve(prob, AutoTsit5(Rosenbrock23()), saveat=traj.t,
          reltol=1e-4, abstol=1e-4,
          adaptive=true,
          sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end

function loss(p, mode)
    total = 0f0
    batch = vcat(rand(data[1:80],2), rand(data[81:100],2))
    for traj in batch
        pred = simulate(traj, p, mode=mode)
        total += mean((pred[1,:] .- traj.h).^2) + 0.1f0*mean((pred[2,:] .- traj.v).^2)
        mask = traj.h .< 0.5f0
        if sum(mask)>0
            total += 2.0f0*mean((pred[1,mask] .- traj.h[mask]).^2)
        end
    end
    total / Float32(length(batch))
end

############################################################
# TRAIN
############################################################

function train(mode)
    mkpath("results")
    if mode=="baseline"
        println("⏩ Skipping Baseline...")
        serialize("results/baseline_weights.jls", ComponentArray(log_scale=0f0, log_decay=0f0, log_beta=0f0, nn=nn_ps))
        flush(stdout)
        return
    end

    println("🚀 Training $mode...")
    flush(stdout)
    p = ComponentArray(log_scale=0f0, log_decay=-1f0, log_beta=0f0, nn=nn_ps)
    opt_state = Optimisers.setup(Optimisers.Adam(1e-3), p)
    loss_log = Float32[]

    for epoch in 1:EPOCHS
        l, back = Zygote.pullback(ps -> loss(ps, mode), p)
        gs = back(one(l))[1]

        for k in keys(gs)
            gs[k] = clamp.(gs[k], -1f0, 1f0)
        end
        opt_state, p = Optimisers.update!(opt_state, p, gs)
        push!(loss_log, l)
        if epoch % 5==0
            println("$mode Epoch $epoch | Loss: $l")
            flush(stdout)
        end
    end
    serialize("results/$(mode)_weights.jls", p)
    serialize("results/$(mode)_loss.jls", loss_log)
end

train("baseline")
train("blackbox")
train("ude")
println("✅ ALL MODELS COMPLETED")
flush(stdout)