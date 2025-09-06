using LinearAlgebra
using ForwardDiff
using Plots  
using Plots: contour, heatmap  # Explicitly import contour

# Objective: y - μ*(log(s1) + log(s2) + log(s3))
function objective(z, μ)
    y, s1, s2, s3 = z
    return y - μ * (log(s1) + log(s2) + log(s3))
end

# Constraints: equality constraints
function constraints(z)
    y, s1, s2, s3 = z
    c1 = (3sqrt(3)/2)*(y^3 - y) - x_val - s1
    c2 = y + 1 - s2
    c3 = y - 1.5 + s3
    return [c1, c2, c3]
end

# ---- 计算 gap 的小工具 ----
function compute_gaps(z, λ, μ)
    # J = ∂c/∂(y,s1,s2,s3)  → 取 s 的三列
    J = ForwardDiff.jacobian(constraints, z)
    JsT = J[:, 2:4]'                # (∂c/∂s)^T
    s = z[2:4]
    w_imp = JsT * λ                 # 隐式对偶
    gap_pd = dot(s, w_imp)          # PD 式 gap (用隐式对偶)
    gap_barrier = length(s) * μ     # 理论 barrier gap = m*μ
    comp_resid = s .* w_imp .- μ    # 互补残差向量
    return gap_pd, gap_barrier, comp_resid
end

# KKT residual function
function kkt_residual(z, λ, μ)
    grad_f = ForwardDiff.gradient(z -> objective(z, μ), z)
    jac_c  = ForwardDiff.jacobian(z -> constraints(z), z)
    c = constraints(z)
    stationarity = grad_f + jac_c' * λ
    return vcat(stationarity, c)
end

# Newton step for (z, λ)
function newton_step(z, λ, μ)
    n = length(z)
    m = length(λ)
    Lxx = ForwardDiff.hessian(z -> objective(z, μ), z)
    J   = ForwardDiff.jacobian(z -> constraints(z), z)
    KKT_mat = [Lxx  J'; J  zeros(m, m)]
    rhs = -kkt_residual(z, λ, μ)
    dzλ = KKT_mat \ rhs
    return dzλ[1:n], dzλ[n+1:end]
end

# Main solver
function solve_barrier_once(z0; μ, max_iter=80, tol=1e-8, verbose=true)
    z = z0
    λ = zeros(3)
    z_hist = [copy(z)]
    for k = 1:max_iter
        res = kkt_residual(z, λ, μ)
        if norm(res) < tol
            println("Converged in $k steps.")
            return z, λ, z_hist
        end
        dz, dλ = newton_step(z, λ, μ)
        α = 1.0
        while any(z .+ α*dz .<= 0)  # Ensure positivity of s1, s2, s3
            α *= 0.5
        end
        z += α * dz
        λ += α * dλ
        push!(z_hist, copy(z))
        gap_pd, gap_barrier, comp_resid = compute_gaps(z, λ, μ)
        println("iter $k AFTER   α = ", α,
                "  gap_PD = ", gap_pd,
                "  gap_bar = ", gap_barrier,
                "  comp_norm = ", norm(comp_resid))
    end
    println("Max iteration reached.")
    return z, λ, z_hist
    
end

function solve_barrier_outer(z0; μ0=0.5, factor=0.3, n_outer=80,
                             inner_max_iter=80, inner_tol=1e-8, verbose=true)
    z = copy(z0)
    λ = zeros(3)
    μ_vals = Float64[]
    z_hist_last = Vector{Vector{Float64}}()  # 保存最后一次内层轨迹

    for t in 0:n_outer-1
        μ = μ0 * factor^t
        push!(μ_vals, μ)
        verbose && println("========== Outer t=$(t), μ=$(μ) ==========")
        z, λ, z_hist = solve_barrier_once(z; μ=μ, max_iter=inner_max_iter, tol=inner_tol, verbose=verbose)
        z_hist_last = z_hist  # 记录最后一次内层的轨迹
    end
    return z, λ, μ_vals, z_hist_last
end



function solve_barrier_filter(z0; max_iter=20, tol=1e-6)
    z = z0
    λ = zeros(3)
    z_hist = [copy(z)]
    γ_f = 0.01
    γ_θ = 0.01
    filter_points = []  # list of tuples (θ, f)

    for k = 1:max_iter
        grad_f = ForwardDiff.gradient(objective, z)
        jac_c = ForwardDiff.jacobian(constraints, z)
        res = kkt_residual(z, λ)
        θ_k = norm(constraints(z))
        f_k = objective(z)

        if norm(res) < tol
            println("Converged in $k steps.")
            return z, λ,z_hist
        end

        dz, dλ = newton_step(z, λ)

        # Fraction-to-the-boundary for s_i > 0
        α_max = 1.0
        for i in 2:4  # s1, s2, s3
            if dz[i] < 0
                α_max = min(α_max, -0.99 * z[i] / dz[i])  # e.g. τ = 0.01
            end
        end

        # Filter line search
        success = false
        for j in 0:20
            α = (1/2)^j * α_max
            z_trial = z + α * dz
            θ_trial = norm(constraints(z_trial))
            f_trial = objective(z_trial)

            # Check filter accept criteria
            f_cond = f_trial <= f_k - γ_f * θ_k
            θ_cond = θ_trial <= θ_k - γ_θ * θ_k

            if f_cond || θ_cond
                # Accept point
                z = z_trial
                λ += α * dλ
                push!(z_hist, copy(z))
                push!(filter_points, (θ_trial, f_trial))
                success = true
                break
            end
        end

        if !success
            println("Filter line search failed at iteration $k.")
            break
        end
    end

    println("Max iteration reached.")
    return z, λ,z_hist
end


function plot_contour_and_path(objective_with_mu, z_hist; xlim=(-1.5,1.5), ylim=(-3,3), nx=300, ny=300, mode=:contour)
    ys  = range(xlim[1], xlim[2], length=nx)
    s1s = range(ylim[1], ylim[2], length=ny)
    Z = Array{Float64}(undef, ny, nx)

    for (i, s1) in enumerate(s1s), (j, y) in enumerate(ys)
        s2 = y + 1
        s3 = 1.5 - y
        if s1 > 0 && s2 > 0 && s3 > 0
            Z[i, j] = objective_with_mu([y, s1, s2, s3])
        else
            Z[i, j] = NaN
        end
    end

    plt = if mode == :heatmap
        heatmap(ys, s1s, Z, xlabel="y", ylabel="s1",
                title="Objective slice + Barrier path", colorbar=true)
    else
        contour(ys, s1s, Z, levels=30, xlabel="y", ylabel="s1",
                title="Objective slice + Barrier path")
    end

    # 投影路径到 (y, s1)
    ypath  = [z[1] for z in z_hist]
    s1path = [z[2] for z in z_hist]
    plot!(plt, ypath, s1path, marker=:o, lw=2, label="Barrier path")
    for i in 1:length(z_hist)
        annotate!(plt, (z_hist[i][1], z_hist[i][2], text("$(i-1)", 8, :black)))
    end
    return plt
end

# Call it:
#z_sol, λ_sol = solve_barrier_filter(z0)
# Problem parameters
const x_val = 3.0

# Initial guess for [y, s1, s2, s3]
z0 = [0.5, 1.0, 1.0, 0.5]  # [y, s1, s2, s3]

z_sol, λ_sol, μ_vals, z_hist_last =
    solve_barrier_outer(z0; μ0=0.5, factor=0.3, n_outer=6,
                        inner_max_iter=30, inner_tol=1e-8, verbose=true)

println("\nFinal Solution z = ", z_sol)
println("Final λ = ", λ_sol)
gap_pd, gap_bar, comp = compute_gaps(z_sol, λ_sol, μ_vals[end])
println("Final gap_PD = ", gap_pd,
        "   barrier gap(mμ) = ", gap_bar,
        "   comp_norm = ", norm(comp))

# 用“最后一个 μ”的目标画等高线与最终轨迹
plt = plot_contour_and_path(z -> objective(z, μ_vals[end]), z_hist_last; mode=:contour)
out = joinpath(pwd(), "barrier_outer_path.png")
savefig(plt, out)
println("Saved figure to: ", out)
display(plt)