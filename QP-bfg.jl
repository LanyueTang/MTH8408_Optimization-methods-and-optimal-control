using LinearAlgebra
using Optim
using Plots
using ForwardDiff

# === Objective ===
function f(x)
    # x = [x1, x2]
    return 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
end

function grad_f(x)
    # ∂f/∂x1 = -400*x1*(x2 - x1^2) - 2*(1 - x1)
    # ∂f/∂x2 =  200*(x2 - x1^2)
    g1 = -400*x[1] * (x[2] - x[1]^2) - 2*(1 - x[1])
    g2 =  200*(x[2] - x[1]^2)
    return [g1, g2]
end

# === Constraints ===
function c(x)
    return [x[1] + x[2] + 10.0]   # equality constraint
end

function g(x)
    return [x[1] + x[2], x[1] - x[2]]  # inequality constraints
end

function J_c()
    return [1.0 1.0]
end

function J_g()
    return [1.0 1.0; 1.0 -1.0]
end

# === Full penalty function ===
function penalty(z, ρ)
    x, s = z[1:2], z[3:4]
    return f(x) + (norm(c(x))^2 + norm(g(x) - s)^2) / (2ρ)
end

function stop_condition(x, s, λ, μ, εa, εr, r0, grad_hist, eq_hist, ineq_hist)
    r = vcat(grad_f(x) - J_c()' * λ - J_g()' * μ, c(x), g(x) - s)
    push!(grad_hist, norm(grad_f(x) - J_c()' * λ - J_g()' * μ))
    push!(eq_hist,   norm(c(x)))
    push!(ineq_hist, norm(g(x) - s))
    flag = norm(r) <= εa + εr * norm(r0)
    return flag, grad_hist, eq_hist, ineq_hist
end

using Plots

function plot_penalty_contour(x, ρ, k, penalty)
    xrange = range(-15, 15, length=200)
    yrange = range(-15, 10, length=200)
    Z = zeros(length(yrange), length(xrange))
    for (i, y) in enumerate(yrange), (j, x1) in enumerate(xrange)
        x_tmp = [x1, y]
        s_tmp = [x1 + y, x1 - y]
        z_tmp = vcat(x_tmp, s_tmp)
        Z[i, j] = penalty(z_tmp, ρ)
    end
    return xrange, yrange, Z
end

function plot_path(path, xrange, yrange, Z, ρ, k)
    contourf(xrange, yrange, Z, levels=30, c=:viridis,
             xlabel="x₁", ylabel="x₂",
             title="Penalty Path at ρ = $ρ (Iter $k)", legend=false)
    path_mat = hcat(path...)
    plot!(path_mat[1, :], path_mat[2, :], lw=2, color=:blue)
    scatter!(path_mat[1, :], path_mat[2, :], color=:white, markersize=4)
    savefig("penalty_path_iter_$k.png")
end
# === 2) 绘图函数，带 r0 参考线 ===
function plot_residuals_with_r0(grad_hist, eq_hist, ineq_hist,
                                r0_grad, r0_eq, r0_iq)
    its = 1:length(grad_hist)
    p = plot(its, grad_hist,
             label="‖∇f−Jᵀλ−Jᵀμ‖",
             xlabel="Outer Iteration", ylabel="Residual Norm",
             lw=2)
    plot!(its, eq_hist,
          label="‖c(x)‖", lw=2, ls=:dash)
    plot!(its, ineq_hist,
          label="‖g(x)−s‖", lw=2, ls=:dot)

    # 加入三条水平线作为 r0 参考
    hline!([r0_grad], label="initial ‖∇f−Jᵀλ−Jᵀμ‖",  lw=1, ls=:dashdot)
    hline!([r0_eq],   label="initial ‖c(x)‖",        lw=1, ls=:dashdot)
    hline!([r0_iq],   label="initial ‖g(x)−s‖",      lw=1, ls=:dashdot)

    title!("KKT Residuals vs. Iteration")
    return p
end

function quadratic_penalty_method()
    z = [10.0, -8.0, -11, 0.0]
    lower = [-100.0, -100.0, -100.0, -100.0]
    upper = [100.0, 100.0, -10.0, 0.2]

    ρ = 1.0
    εa, εr = 1e-6, 1e-6
    εa_k, εr_k = 1e-2, 1e-2
    max_iter = 8

    path = [copy(z)]
    λ = -1/ρ * c(z[1:2])
    μ = -1/ρ * (g(z[1:2]) - z[3:4])
    λ0, μ0, x0, s0 = copy(λ), copy(μ), copy(z[1:2]), copy(z[3:4])
    # 计算初始残差三项
    r0_grad = norm(grad_f(x0) .- J_c()'*λ0 .- J_g()'*μ0)
    r0_eq   = norm(c(x0))
    r0_iq   = norm(g(x0) .- s0)
    r0 = vcat(r0_grad, r0_eq, r0_iq)
    # === 1) 历史数据容器 ===
    grad_hist = Float64[]
    eq_hist   = Float64[]
    ineq_hist = Float64[]
    for k in 1:max_iter
        println("Penalty iteration $k with ρ = $ρ")
        println("Current z: ", z)
        obj(z) = penalty(z, ρ)

        result = optimize(
            obj,
            lower,
            upper,
            z,
            Fminbox(BFGS()),
            Optim.Options(g_tol = 1e-3, iterations = 100, show_trace = true)
        )
        z = Optim.minimizer(result)
        println("Optimized z: ", z)
        push!(path, copy(z))

        x, s = z[1:2], z[3:4]

        # 正确解构 stop_condition 的返回值
        flag, grad_hist, eq_hist, ineq_hist = stop_condition(x, s, λ, μ, εa, εr, r0, grad_hist, eq_hist, ineq_hist)

        if flag  # 只使用布尔标志
            println("✅ Converged at iteration $k")
            break
        end

        λ = -1/ρ * c(x)
        μ = -1/ρ * (g(x) - s)
        #λ0, μ0, x0, s0 = copy(λ), copy(μ), copy(x), copy(s)
        xrange, yrange, Z = plot_penalty_contour(x, ρ, k, penalty)
        plot_path(path, xrange, yrange, Z, ρ, k)

        εa_k /= 2; εr_k /= 2; ρ /= 2
    end
    p = plot_residuals_with_r0(grad_hist, eq_hist, ineq_hist,
                          r0_grad, r0_eq, r0_iq)
    savefig("residuals_vs_r0.png")
    display(p)
    return z[1:2], z[3:4]
end

x_sol, s_sol = quadratic_penalty_method()

@show x_sol, s_sol
