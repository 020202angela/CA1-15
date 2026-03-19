############################################################
# ULTIMATE DIAGNOSTIC DASHBOARD (PAPER LEVEL - FINAL FIXED)
############################################################

using Serialization, ComponentArrays
using Plots, Statistics, CSV, DataFrames, Interpolations
using DifferentialEquations, Lux, Random

gr()

const M, g = 1.5f0, 9.81f0

############################################################
# LOAD DATA
############################################################

function load_data(idx=85)
    files = sort(readdir("data", join=true))
    f = files[idx]
    println("📊 Loading analysis target: ", f)
    df = CSV.read(f, DataFrame)
    return (t=Float32.(df.time),
            h=Float32.(df.altitude),
            v=Float32.(df.velocity),
            itp=LinearInterpolation(Float32.(df.time),
                                    Float32.(df.thrust),
                                    extrapolation_bc=Interpolations.Flat()))
end

traj = load_data(85)

############################################################
# LOAD MODELS & WEIGHTS
############################################################

p_black = deserialize("results/blackbox_weights.jls")
p_ude   = deserialize("results/ude_weights.jls")
l_black_log = deserialize("results/blackbox_loss.jls")
l_ude_log   = deserialize("results/ude_loss.jls")

nn = Lux.Chain(Lux.Dense(2=>64, tanh),
               Lux.Dense(64=>64, tanh),
               Lux.Dense(64=>1))
_, st = Lux.setup(Random.default_rng(42), nn)

softplus(x) = log1p(exp(clamp(x, -20f0, 20f0)))

############################################################
# FORCE MODEL
############################################################

function get_ge(h, v, p, mode)
    if mode == "baseline" return 0f0 end
    if mode == "blackbox"
        out, _ = nn(vcat(h, v), p.nn, st)
        return 10f0 * tanh(out[1])
    end
    # UDE
    scale, decay, beta = softplus(p.log_scale), softplus(p.log_decay), softplus(p.log_beta)
    gate = 1f0 / (1f0 + exp(clamp((h - 0.5f0) / 0.1f0, -50f0, 50f0)))
    f_phys = scale * exp(-decay * h) * (1 + beta * v^2)
    out, _ = nn(vcat(h, v), p.nn, st)
    return clamp(gate * (f_phys + 10f0 * tanh(out[1])), 0f0, 60f0)
end

############################################################
# SIMULATION
############################################################

function sim(traj, p, mode)
    function rhs!(du, u, _, t)
        F = get_ge(u[1], u[2], p, mode)
        du[1] = -u[2]
        du[2] = g - (traj.itp(t) + 0.8f0 * u[2] + F) / M
    end
    sol = solve(ODEProblem(rhs!, [traj.h[1], traj.v[1]], (traj.t[1], traj.t[end])),
                AutoTsit5(Rosenbrock23()), saveat=traj.t)
    Array(sol)
end

println("⏳ Simulating all variants...")
base  = sim(traj, nothing, "baseline")
black = sim(traj, p_black, "blackbox")
ude   = sim(traj, p_ude, "ude")

############################################################
# PLOT GENERATION
############################################################

println("🎨 Generating 10-panel dashboard...")

# 1. Trajectory
p1 = plot(traj.t, traj.h, label="Ground Truth", lw=3, color=:black)
plot!(p1, traj.t, base[1,:], label="White-box", ls=:dash, color=:red, alpha=0.7)
plot!(p1, traj.t, black[1,:], label="Pure Black-box", ls=:dot, color=:green, lw=1.5)
plot!(p1, traj.t, ude[1,:], label="Grey-box UDE", lw=2, color=:blue)
xlabel!(p1, "Time (s)"); ylabel!(p1, "Altitude (m)"); title!(p1, "Landing Trajectory (Full)")

# 2. Near-ground Zoom
mask = traj.h .< 0.5f0
idx = findall(mask)
p2 = plot(traj.t[idx], traj.h[idx], label="GT", lw=3, color=:black)
plot!(p2, traj.t[idx], base[1,idx], label="White", ls=:dash, color=:red, alpha=0.7)
plot!(p2, traj.t[idx], black[1,idx], label="Black", ls=:dot, color=:green, lw=1.5)
plot!(p2, traj.t[idx], ude[1,idx], label="UDE", lw=2, color=:blue)
xlabel!(p2, "Time (s)"); ylabel!(p2, "Altitude (m)"); title!(p2, "Near-ground Zoom (h < 0.5m)")

# 3. Velocity
p3 = plot(traj.t, traj.v, label="GT", lw=3, color=:black)
plot!(p3, traj.t, base[2,:], label="White", ls=:dash, color=:red, alpha=0.7)
plot!(p3, traj.t, black[2,:], label="Black", ls=:dot, color=:green, lw=1.5)
plot!(p3, traj.t, ude[2,:], label="UDE", lw=2, color=:blue)
xlabel!(p3, "Time (s)"); ylabel!(p3, "Velocity (m/s)"); title!(p3, "Velocity Profile Comparison")

# 4. Log-scale Error
err_w = abs.(base[1,:] .- traj.h) .+ 1e-6
err_b = abs.(black[1,:] .- traj.h) .+ 1e-6
err_u = abs.(ude[1,:] .- traj.h) .+ 1e-6
p4 = plot(traj.t, err_w, yaxis=:log, label="White", color=:red, alpha=0.5)
plot!(p4, traj.t, err_b, yaxis=:log, label="Black", color=:green, ls=:dot)
plot!(p4, traj.t, err_u, yaxis=:log, label="UDE", color=:blue, lw=2)
xlabel!(p4, "Time (s)"); ylabel!(p4, "Abs Error (m)"); title!(p4, "Error Propagation (Log-Scale)")

# 5. Phase Portrait
p5 = plot(traj.h, traj.v, label="GT", lw=3, color=:black)
plot!(p5, base[1,:], base[2,:], label="White", ls=:dash, color=:red, alpha=0.7)
plot!(p5, black[1,:], black[2,:], label="Black", ls=:dot, color=:green, lw=1.5)
plot!(p5, ude[1,:], ude[2,:], label="UDE", lw=2, color=:blue)
xlabel!(p5, "Altitude (m)"); ylabel!(p5, "Velocity (m/s)"); title!(p5, "Phase Portrait Analysis")

# 6. Energy Analysis
E_func(x, v) = M * g * x .+ 0.5f0 * M * v.^2
p6 = plot(traj.t, E_func(traj.h, traj.v), label="GT", lw=3, color=:black)
plot!(p6, traj.t, E_func(base[1,:], base[2,:]), label="White", ls=:dash, color=:red, alpha=0.7)
plot!(p6, traj.t, E_func(black[1,:], black[2,:]), label="Black", ls=:dot, color=:green, lw=1.5)
plot!(p6, traj.t, E_func(ude[1,:], ude[2,:]), label="UDE", lw=2, color=:blue)
xlabel!(p6, "Time (s)"); ylabel!(p6, "Energy (J)"); title!(p6, "Mechanical Energy Decay")

# 7. Heatmap
h_rng, v_rng = 0:0.05:2.5, 0:0.05:2.0
Z = [get_ge(h, v, p_ude, "ude") for h in h_rng, v in v_rng]
p7 = heatmap(h_rng, v_rng, Z', title="UDE: Learned GE Force Field", xlabel="Altitude (m)", ylabel="Velocity (m/s)", color=:turbo)

# 8. Global RMSE
rmse(a, b) = sqrt(mean((a .- b).^2))
p8 = bar(["White", "Black", "UDE"],
    [rmse(base[1,:], traj.h), rmse(black[1,:], traj.h), rmse(ude[1,:], traj.h)],
    title="Global RMSE (Altitude)", color=[:red, :green, :blue], legend=false, ylabel="RMSE (m)")

# 9. Near-ground RMSE
p9 = bar(["White", "Black", "UDE"],
    [rmse(base[1,mask], traj.h[mask]), rmse(black[1,mask], traj.h[mask]), rmse(ude[1,mask], traj.h[mask])],
    title="Near-ground RMSE (h < 0.5m)", color=[:red, :green, :blue], legend=false, ylabel="RMSE (m)")

# 10. Training Stability
p10 = plot(l_black_log, yaxis=:log, label="Pure Black-box", color=:green, alpha=0.5)
plot!(p10, l_ude_log, yaxis=:log, label="Grey-box UDE", color=:blue, lw=1.5)
xlabel!(p10, "Training Epochs"); ylabel!(p10, "Log Loss"); title!(p10, "Convergence Stability")

############################################################
# SAVE
############################################################

final_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
    layout=(5, 2), size=(1500, 2400), margin=14Plots.mm)

savefig("results/final_dashboard_paper_ready.png")
println("✅ SUCCESS: Advanced Dashboard saved to results/final_dashboard_paper_ready.png")