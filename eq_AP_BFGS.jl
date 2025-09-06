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

function J_c()
    return [1.0 1.0]
end

function J_h(z)
    return [
        1.0   1.0    0.0   0.0
    ]
end


function ∇L_A_eq(x, y, ρ)
    grad_f_x = grad_f(x)            # 2×1
    J = J_c()                       # 1×2
    λ = y .+ c(x) .* ρ              # scalar + scalar → scalar
    return grad_f_x .+ J' * λ       # 2×1
end

function make_obj_and_grad_eq!(y, ρ)
    obj(x) = L_A_eq(x, y, ρ)
    function grad!(G, x)
        G .= ∇L_A_eq(x, y, ρ)
        return nothing
    end
    return obj, grad!
end


# === 2) 增广拉格朗日 & 梯度 ===

function L_A_eq(x, y, ρ)
    return f(x) + y[1] * c(x)[1] + (ρ/(2)) * c(x)[1]^2
end



# === 3) 停止准则 & 残差记录 ===
function record_and_stop(x, y, εa, εr, r0, grad_hist, eq_hist)
    # KKT gradient residual: ∇f(x) - J_cᵀ * y
    grad_L = grad_f(x) .+ J_c()' * y
    r_grad = norm(grad_L)
    r_eq   = norm(c(x))

    # 历史记录
    push!(grad_hist, r_grad)
    push!(eq_hist, r_eq)

    # 整体残差向量（用于整体终止判断）
    r = [grad_L; r_eq]

    # 子问题终止条件（通常用于 ALM 中的内层解算器）
    flag_sub = r_eq ≤ εa + εr * norm(r0)

    # 返回整体终止、子问题终止标志和残差历史
    return norm(r) ≤ εa + εr*norm(r0), flag_sub, grad_hist, eq_hist
end

function plot_penalty_contour(x, ρ, k, penalty)
    xrange = range(0, 20, length=200)
    yrange = range(-10, 100, length=200)
    Z = zeros(length(yrange), length(xrange))
    for (i, y) in enumerate(yrange), (j, x1) in enumerate(xrange)
        x_tmp = [x1, y]
        Z[i, j] = penalty(x_tmp, ρ)
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
    savefig("/Users/lanyuetang/Documents/learning/MATH8408/0618增广拉格朗日/figure/AP/eq/penalty_path_iter_$k.png")
end
function plot_residuals_with_r0(grad_hist, eq_hist, r0_grad, r0_eq, εa, εr)
    its = 1:length(grad_hist)

    # 计算参考值
    e_r0_grad = εa + εr * norm(r0_grad)
    e_r0_eq   = εa + εr * norm(r0_eq)

    # 主图
    p = plot(its, grad_hist,
             label="‖∇f+Jᵀλ‖",
             xlabel="Outer Iteration", ylabel="Residual Norm",
             lw=2, color=:red, ylims=(0, 2))
    plot!(its, eq_hist,
          label="‖c(x)‖", lw=2, color=:green, ylims=(0, 2))

    # 添加参考线
    hline!([e_r0_grad], label="initial ‖∇f+Jᵀλ‖", lw=1, ls=:dashdot, color=:red)
    hline!([e_r0_eq],   label="initial ‖c(x)‖",     lw=1, ls=:dashdot, color=:green)

    # 标题
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
        legend = :topleft)
    return p
end
function plot_y_history(y_hist::Vector{Float64})
    iters = 1:length(y_hist)
    p = plot(iters, y_hist,
        xlabel = "Outer Iteration",
        ylabel = "Multiplier y",
        lw = 2,
        marker = :circle,
        label = "y",
        title = "Multiplier y vs. Iteration",
        legend = :topleft)
    return p
end


# === 4) 外层 ALM ===
function alm_with_slack()
    # init
    z = [10.0, -8.0]      # [x; s]
    y = zeros(1)                      # multipliers for [c; g−s]
    ρ = 1; q = 0.5
    εa, εr = 1e-7, 1e-7
    εa_k, εr_k = 1e-2, 1e-2
    max_iter = 100

    # 初始残差 r0
    r0 = vcat(
      grad_f(z[1:2]) .+ J_c()'*y[1:1],
      c(z[1:2])
    )

    # history & path
    grad_hist = Float64[]
    eq_hist = Float64[]
    ρ_hist = Float64[]
    y_hist = Float64[]
    for k in 1:max_iter
        println("ALM iter $k, ρ=$ρ")
        push!(ρ_hist, ρ)
        push!(y_hist, copy(y[1]))
        # — 内层子问题，用无约束 BFGS 解增广拉格朗日 —
        println("Current z: ", z)
        # ——— 1) 计算内层停止阈值 ———
        g0 = ∇L_A_eq(z, y, ρ)
        inner_tol = εa_k + εr_k * norm(g0)
        inner_path = Vector{Vector{Float64}}()
        push!(inner_path, copy(z[1:2]))
        obj(z) = L_A(z, y, ρ)
        ∇obj(z) = ∇L_A_eq(z, y, ρ)
        obj, grad! = make_obj_and_grad_eq!(y, ρ)
        od = OnceDifferentiable(obj, grad!, copy(z))  # copy(z) 防止被覆盖

        opt = optimize(
            od,
            z,
            BFGS(),
            Optim.Options(g_tol = max(inner_tol, 1e-6), 
            iterations = 100, 
            show_trace = false, 
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

    
        println("  x=$(z[1:2])")

        # — record & stop?
        flag, flag_sub, grad_hist, eq_hist =
           record_and_stop(z, y, εa, εr, r0,
                           grad_hist, eq_hist)
        if flag

            
            println("Converged at iter $k")
            println("Final r: ")
            print(r0)
            println("Final grad: ")
            print(grad_hist, eq_hist)
            break
        end

        # — 同原逻辑画图 & 残差图 —
        xrange, yrange, Z = plot_penalty_contour(
            z[1:2], ρ, k,
            (x_, ρ_) -> L_A_eq(x_, y, ρ_)
        )

        plot_path(inner_path, xrange, yrange, Z, ρ, k)
        # — 按板书更新 ρ, εa_k, εr_k, y —

        #if  k%2 ==0 || flag_sub || norm(c(z[1:2])) ≤ q*norm(c(inner_path[end-1]))
        if  norm(c(z[1:2])) ≤ q*norm(c(inner_path[end-1]))    
        # keep ρ, ε’s, update y
            println("  Updating y with current z")
            # slack variables
            y .= y .+ c(z[1:2]) .* ρ
        else
            println("Decreasing ρ, εa_k, εr_k")
            εa_k *= 0.1; εr_k *= 0.1
            ρ    *= 5
            # 这里可以加入对 ρ 的下限检测
        end


    end

    # — 最后画残差历史图，加上 r0 参考线 —
    p = plot_residuals_with_r0(grad_hist, eq_hist, 
                                r0[1:2], r0[3], εa, εr)
                                
    savefig("/Users/lanyuetang/Documents/learning/MATH8408/0618增广拉格朗日/figure/AP/eq/alm_residuals.png"); display(p)
    p2 = plot_rho_history(ρ_hist)
    savefig("/Users/lanyuetang/Documents/learning/MATH8408/0618增广拉格朗日/figure/AP/eq/rho_vs_iteration.png")
    display(p2)
    println("Final y: ", y_hist)
    p3 = plot_y_history(y_hist)
    savefig("/Users/lanyuetang/Documents/learning/MATH8408/0618增广拉格朗日/figure/AP/eq/y_vs_iteration.png")
    display(p3)


    return z[1:2]
end

x_star = alm_with_slack()
@show x_star
