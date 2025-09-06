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
    return [x[1] + x[2] - 1.0]   # equality constraint
end

function g(x)
    return [-x[1] - 2*x[2] + 8 , -x[1] + x[2] - 0.5]  # inequality constraints
end

function J_c()
    return [1.0 1.0]
end

function J_g()
    return [-1.0 -2.0; -1.0 1.0]
end

function J_h(z)
    x = z[1:2]; t = z[3:4]
    σ = @. exp(t) / (1 + exp(t))  # sigmoid(t)
    return [
        1.0   1.0    0.0          0.0;
       -1.0  -2.0   -σ[1]         0.0;
       -1.0   1.0    0.0         -σ[2]
    ]
end
function ∇L_A_manual(z, y, ρ)
    x = z[1:2]; t = z[3:4]
    s = log1p.(exp.(t))  # slack variables
    h = vcat(c(x), g(x) .- s)
    λ = y + h / ρ        # effective multiplier

    grad_f_x = grad_f(x)            # Rosenbrock gradient
    J = J_h(z)                      # 3×4 Jacobian
    grad_LA = zeros(4)
    grad_LA[1:2] .= grad_f_x .+ J[:, 1:2]' * λ   # ∇ₓ part
    grad_LA[3:4] .= J[:, 3:4]' * λ               # ∇ₜ part
    return grad_LA
end

function make_obj_and_grad!(y, ρ)
    obj(z) = L_A(z, y, ρ)
    function grad!(G, z)
        G .= ∇L_A_manual(z, y, ρ)  # 写入 G 中
        return nothing
    end
    return obj, grad!
end


# === 2) 增广拉格朗日 & 梯度 ===
"""
 z = [x; s],   s∈R² slack for g
 h(z) = [ c(x) ;  g(x) - s ] ∈ R^3
 y ∈ R^3 multipliers
"""
function L_A(z, y, ρ)
    x = z[1:2]; t = z[3:4]
    s = log1p.(exp.(t))  # slack variables
    h = vcat(c(x), g(x) .- s)
    return f(x) + dot(y, h) + norm(h)^2/(2ρ)
end


# === 3) 停止准则 & 残差记录 ===
function record_and_stop(z, y, εa, εr, r0, grad_hist, eq_hist, iq_hist)
    x, t = z[1:2], z[3:4]
    s = log1p.(exp.(t))  # slack variables
    # KKT‐gradient residual = ∇f - J_h^T y
    gh = vcat(grad_f(x), zeros(2)) .- J_h(z)' * y
    r_grad = norm(gh)
    r_eq   = norm(c(x))
    r_iq   = norm(g(x).-s)
    push!(grad_hist, r_grad)
    push!(eq_hist,   r_eq)
    push!(iq_hist,   r_iq)
    r = [gh; r_eq; r_iq]
    flag_sub = (r_eq ≤ εa + εr * norm(r0)) || (r_iq ≤ εa + εr * norm(r0))
    return norm(r) ≤ εa + εr*norm(r0), flag_sub, grad_hist, eq_hist, iq_hist
end
function plot_penalty_contour(x, ρ, k, penalty)
    xrange = range(0, 5, length=200)
    yrange = range(0, 5, length=200)
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
             lw=2,color=:red, ylims=(0, 1000))
    plot!(its, eq_hist,
          label="‖c(x)‖", lw=2, color=:green, ylims=(0, 1000))
    plot!(its, ineq_hist,
          label="‖g(x)−s‖", lw=2, color=:blue, ylims=(0, 1000))
    e_r0_grad = εa + εr*norm(r0_grad)
    e_r0_eq   = εa + εr*norm(r0_eq)
    e_r0_iq   = εa + εr*norm(r0_iq)
    # 加入三条水平线作为 r0 参考
    hline!([e_r0_grad], label="initial ‖∇f−Jᵀλ−Jᵀμ‖",  lw=1, ls=:dashdot, color =:red)
    hline!([e_r0_eq],   label="initial ‖c(x)‖",        lw=1, ls=:dashdot, color =:green)
    hline!([e_r0_iq],   label="initial ‖g(x)−s‖",      lw=1, ls=:dashdot, color =:blue)

    title!("KKT Residuals vs. Iteration")
    return p
end
function plot_rho_history(ρ_hist::Vector{Float64})
    iters = 1:length(ρ_hist)
    p = plot(iters, ρ_hist,
        xlabel = "Outer Iteration",
        ylabel = "Penalty ρ",
        yscale = :log10,
        lw = 2,
        marker = :circle,
        label = "ρ",
        title = "Penalty Parameter ρ vs. Iteration",
        legend = :topright)
    return p
end


# === 4) 外层 ALM ===
function alm_with_slack()
    # init
    z = [10.0, -8.0, 0.1, 0.2]      # [x; s]
    y = zeros(3)                      # multipliers for [c; g−s]
    ρ = 10.0; q = 1
    εa, εr = 1e-6, 1e-6
    εa_k, εr_k = 1e-2, 1e-2
    max_iter = 200

    # 初始残差 r0
    x0, t0 = z[1:2], z[3:4]
    s0 = log1p.(exp.(t0)) # slack variables
    r0 = vcat(
      grad_f(x0) .- J_c()'*y[1:1] .- J_g()'*y[2:3],
      c(x0),
      g(x0).-s0
    )

    # history & path
    grad_hist = Float64[]; eq_hist = Float64[]; iq_hist = Float64[]
    ρ_hist = Float64[]
    for k in 1:max_iter
        println("ALM iter $k, ρ=$ρ")
        push!(ρ_hist, ρ)
        # — 内层子问题，用无约束 BFGS 解增广拉格朗日 —
        println("Current z: ", z)
        # ——— 1) 计算内层停止阈值 ———
        g0 = ∇L_A_manual(z, y, ρ)
        inner_tol = εa_k + εr_k * norm(g0)
        inner_path = Vector{Vector{Float64}}()
        push!(inner_path, copy(z[1:2]))
        obj(z) = L_A(z, y, ρ)
        ∇obj(z) = ∇L_A_manual(z, y, ρ)
        obj, grad! = make_obj_and_grad!(y, ρ)
        od = OnceDifferentiable(obj, grad!, copy(z))  # copy(z) 防止被覆盖

        opt = optimize(
            od,
            z,
            BFGS(),
            Optim.Options(g_tol = max(inner_tol, 1e-6), 
            iterations = 100, 
            show_trace = true, 
            store_trace = true,
            extended_trace = true,
            )
            ) # Full trace for path)
        # 2) 拿到 trace 中每一个 state 的 x：
        for st in Optim.trace(opt)
            if haskey(st.metadata, "x")
                push!(inner_path, st.metadata["x"][1:2])
            end
        end

        # 3) 最终解
        z = Optim.minimizer(opt)

    
        println("  x=$(z[1:2]), t=$(z[3:4])")

        # — record & stop?
        flag, flag_sub, grad_hist, eq_hist, iq_hist =
           record_and_stop(z, y, εa, εr, r0,
                           grad_hist, eq_hist, iq_hist)
        if flag

            
            println("Converged at iter $k")
            println("Final r: ")
            print(r0)
            println("Final grad: ")
            print(grad_hist, eq_hist, iq_hist)
            break
        end
        # — 同原逻辑画图 & 残差图 —
        xrange,yrange,Z = plot_penalty_contour(z[1:2],ρ,k, (x_,ρ_)->L_A(vcat(x_,z[3:4]),y,ρ_))
        plot_path(inner_path, xrange, yrange, Z, ρ, k)
        # — 按板书更新 ρ, εa_k, εr_k, y —

        if flag_sub || norm(c(z[1:2])) ≤ q*norm(c(inner_path[end-1])) || norm(g(z[1:2]) .- z[3:4]) ≤ q*norm(g(inner_path[end-1]) .- inner_path[end])
            # keep ρ, ε’s, update y
            println("  Updating y with current z")
            t = z[3:4]
            s = log1p.(exp.(t))  # slack variables
            h = vcat(c(z[1:2]), g(z[1:2]) .- s)

            y .= y .+ h/ρ
        else
            println("Decreasing ρ, εa_k, εr_k")   
            εa_k *= 0.1; εr_k *= 0.1
            ρ    *= 0.5
            # 这里可以加入对 ρ 的下限检测
        end


    end

    # — 最后画残差历史图，加上 r0 参考线 —
    p = plot_residuals_with_r0(grad_hist, eq_hist, iq_hist,
                               r0[1], r0[2], r0[3], εa, εr)
    savefig("alm_residuals.png"); display(p)
    p2 = plot_rho_history(ρ_hist)
    savefig("rho_vs_iteration.png")
    display(p2)


    return z[1:2], z[3:4]
end

x_star, t_star = alm_with_slack()
@show x_star, t_star
