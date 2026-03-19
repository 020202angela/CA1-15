############################################################
# UDE SMOKE TEST (PRE-FLIGHT CHECK)
############################################################

include("train_ude.jl") 

println("🔍 Starting Smoke Test...")

const TEST_EPOCHS = 2
const TEST_BATCH = 1

function smoke_test(mode)
    println("Testing mode: $mode")
    try
        p = ComponentArray(log_scale=0f0, log_decay=-1f0, log_beta=0f0, nn=nn_ps)
        opt_state = Optimisers.setup(Optimisers.Adam(1e-3), p)
        
        for epoch in 1:TEST_EPOCHS

            l, back = Zygote.pullback(ps -> loss(ps, mode), p)
            gs = back(one(l))[1]
            
            if any(isnan, gs)
                error("❌ NaN detected in gradients at Epoch $epoch!")
            end
            
            opt_state, p = Optimisers.update!(opt_state, p, gs)
            println("  Epoch $epoch: Loss = $l (Success)")
        end
        println("✅ Mode $mode passed smoke test.")
    catch e
        println("❌ Mode $mode FAILED!")
        rethrow(e)
    end
end

println("--- Initializing Data ---")

println("--- Running Tests ---")
smoke_test("blackbox")
smoke_test("ude")

println("\n✨ ALL TESTS PASSED. Ready for qsub!")