using LinearAlgebra, Optim, ForwardDiff, Plots

# === 1) 目标 & 约束 ===
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

# === 2) 增广拉格朗日 & 梯度 ===
"""
 z = [x; s],   s∈R² slack for g
 h(z) = [ c(x) ;  g(x) - s ] ∈ R^3
 y ∈ R^3 multipliers
"""
function L_A(z, y, ρ)
    x = z[1:2]; s = z[3:4]
    h = vcat(c(x), g(x) .- s)
    return f(x) + dot(y, h) + norm(h)^2/(2ρ)
end

function ∇L_A(z, y, ρ)
    ForwardDiff.gradient(w->L_A(w, y, ρ), z)
end

# === 3) 停止准则 & 残差记录 ===
function record_and_stop(z, y, εa, εr, r0, grad_hist, eq_hist, iq_hist)
    x, s = z[1:2], z[3:4]
    # KKT‐gradient residual = ∇f - J_h^T y
    gh = grad_f(x) .- J_c()'*y[1:1] .- J_g()'*y[2:3]
    r_grad = norm(gh)
    r_eq   = norm(c(x))
    r_iq   = norm(g(x).-s)
    push!(grad_hist, r_grad)
    push!(eq_hist,   r_eq)
    push!(iq_hist,   r_iq)
    r = [gh; r_eq; r_iq]
    return norm(r) ≤ εa + εr*norm(r0), grad_hist, eq_hist, iq_hist
end
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
function plot_residuals_with_r0(grad_hist, eq_hist, ineq_hist,
                                r0_grad, r0_eq, r0_iq, εa, εr)
    its = 1:length(grad_hist)
    p = plot(its, grad_hist,
             label="‖∇f−Jᵀλ−Jᵀμ‖",
             xlabel="Outer Iteration", ylabel="Residual Norm",
             lw=2)
    plot!(its, eq_hist,
          label="‖c(x)‖", lw=2, ls=:dash)
    plot!(its, ineq_hist,
          label="‖g(x)−s‖", lw=2, ls=:dot)
    e_r0_grad = εa + εr*norm(r0_grad)
    e_r0_eq   = εa + εr*norm(r0_eq)
    e_r0_iq   = εa + εr*norm(r0_iq)
    # 加入三条水平线作为 r0 参考
    hline!([e_r0_grad], label="initial ‖∇f−Jᵀλ−Jᵀμ‖",  lw=1, ls=:dashdot)
    hline!([e_r0_eq],   label="initial ‖c(x)‖",        lw=1, ls=:dashdot)
    hline!([e_r0_iq],   label="initial ‖g(x)−s‖",      lw=1, ls=:dashdot)

    title!("KKT Residuals vs. Iteration")
    return p
end
# === 4) 外层 ALM ===
function alm_with_slack()
    # init
    z = [10.0, -8.0, -11.0, 0.0] 
    lower = [-100.0, -100.0, -100.0, -100.0]
    upper = [100.0, 100.0, -9.0, 0.2]      # [x; s]
    y = zeros(3)                      # multipliers for [c; g−s]
    ρ = 1.0; q = 0.2
    εa, εr = 1e-6, 1e-6
    εa_k, εr_k = 1e-2, 1e-2
    max_iter = 20

    # 初始残差 r0
    x0, s0 = z[1:2], z[3:4]
    r0 = vcat(
      grad_f(x0) .- J_c()'*y[1:1] .- J_g()'*y[2:3],
      c(x0),
      g(x0).-s0
    )

    # history & path
    grad_hist = Float64[]; eq_hist = Float64[]; iq_hist = Float64[]
    path = [copy(z[1:2])]

    for k in 1:max_iter
        println("ALM iter $k, ρ=$ρ")
        # — 内层子问题，用无约束 BFGS 解增广拉格朗日 —
        println("Current z: ", z)
                # ——— 1) 计算内层停止阈值 ———
        g0 = ForwardDiff.gradient(w->L_A(w,y,ρ), z)
        inner_tol = εa_k + εr_k * norm(g0)
        obj(z) = L_A(z, y, ρ)
        result = optimize(
            obj,
            lower,
            upper,
            z,
            Fminbox(BFGS()),
            Optim.Options(g_tol = inner_tol, iterations = 100, show_trace = true)
        )
        z = Optim.minimizer(result)
        push!(path, copy(z[1:2]))
        println("  x=$(z[1:2]), s=$(z[3:4])")

        # — record & stop?
        flag, grad_hist, eq_hist, iq_hist =
           record_and_stop(z, y, εa, εr, r0,
                           grad_hist, eq_hist, iq_hist)
        if flag
            println("Converged at iter $k")
            break
        end
        # — 同原逻辑画图 & 残差图 —
        xrange,yrange,Z = plot_penalty_contour(z[1:2],ρ,k, (x_,ρ_)->L_A(vcat(x_,z[3:4]),y,ρ_))
        plot_path(path, xrange, yrange, Z, ρ, k)
        # — 按板书更新 ρ, εa_k, εr_k, y —
        # 先检测 ||c(x_{k+1})||
        if norm(c(z[1:2])) ≤ q*norm(c(path[end-1]))
            # keep ρ, ε’s, update y
            println("  Updating y with current z")
            h = vcat(c(z[1:2]), g(z[1:2]) .- z[3:4])
            y .= y .+ h/ρ
        else
            # decrease ε’s, decrease ρ, keep y
            εa_k *= 0.1; εr_k *= 0.1
            ρ    *= 0.5
        end


    end

    # — 最后画残差历史图，加上 r0 参考线 —
    p = plot_residuals_with_r0(grad_hist, eq_hist, iq_hist,
                               r0[1], r0[2], r0[3], εa, εr)
    savefig("alm_residuals.png"); display(p)

    return z[1:2], z[3:4]
end

x_star, s_star = alm_with_slack()
@show x_star, s_star
