# Updated Quadratic Penalty Method with Newton's Method + highly oscillatory constraint to induce instability
using LinearAlgebra
using Plots

# Objective: Rosenbrock
function f(x)
    return 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end

grad_f(x) = [
    -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2)
]

function hess_f(x)
    h11 = 1200 * x[1]^2 - 400 * x[2] + 2
    h12 = -400 * x[1]
    return [h11 h12; h12 200.0]
end

# Constraint: equality with oscillation (stay unchanged)
function c_eq(x)
    return [sin(3x[1]) + cos(3x[2]) - 0.5]
end

function J_c(x)
    return [3cos(3x[1])  -3sin(3x[2])]
end

# Inequality: pathological log-oscillatory constraint
function g_ineq(x)
    ϵ = 1e-8
    val = abs(sin(10x[1]) + sin(10x[2]))
    return [-log(ϵ + val) + 1]
end

function J_g(x)
    ϵ = 1e-8
    val = abs(sin(10x[1]) + sin(10x[2]))
    sign_term = sign(sin(10x[1]) + sin(10x[2]))
    dval_dx1 = 10cos(10x[1])
    dval_dx2 = 10cos(10x[2])
    dlog_dx1 = -sign_term * dval_dx1 / (ϵ + val)
    dlog_dx2 = -sign_term * dval_dx2 / (ϵ + val)
    return [dlog_dx1 dlog_dx2]
end

function penalty(x, s, ρ)
    ceq = c_eq(x)
    gval = g_ineq(x)
    return f(x) + (norm(ceq)^2 + norm(gval .- s)^2) / (2ρ)
end

function grad_penalty(x, s, ρ)
    ceq = c_eq(x)
    gval = g_ineq(x)
    ∇x = grad_f(x) + (J_c(x)' * ceq + J_g(x)' * (gval .- s)) / ρ
    ∇s = -1/ρ .* (gval .- s)
    return [∇x; ∇s]
end

function hess_penalty(x, s, ρ)
    Hf = hess_f(x)
    Jc = J_c(x)
    Jg = J_g(x)
    Hx = Hf + (Jc' * Jc + Jg' * Jg) / ρ
    return [Hx zeros(2,1); zeros(1,2) 1/ρ]
end

function newton_optimizer(x0, s0, ρ; max_iter=50, tol=1e-6)
    z = [x0; s0]
    path = [copy(z)]
    for _ in 1:max_iter
        g = grad_penalty(z[1:2], z[3:3], ρ)
        H = hess_penalty(z[1:2], z[3:3], ρ)
        if norm(g) < tol
            break
        end
        d = -H \ g
        z += d
        push!(path, copy(z))
    end
    return z[1:2], z[3:3], path
end

function plot_penalty_contour(x, ρ, k; penalty=penalty)
    xrange = range(-2, 2, length=200)
    yrange = range(-2, 2, length=200)
    Z = [penalty([xv, yv], g_ineq([xv, yv]), ρ) for yv in yrange, xv in xrange]
    return Z
end

function plot_bfgs_path_on_penalty(ρ, k, path, Z)
    xgrid = range(-2, 2, length=200)
    ygrid = range(-2, 2, length=200)
    path_mat = hcat(path...)
    contourf(xgrid, ygrid, Z, levels=30, c=:viridis, xlabel="x₁", ylabel="x₂",
             title="Newton Path at ρ = \$ρ (Iter \$k)", legend=false)
    plot!(path_mat[1, :], path_mat[2, :], lw=2, color=:blue, label="Newton Path")
    scatter!(path_mat[1, :], path_mat[2, :], color=:white, label="", markersize=4)
    savefig("newton_path_with_contour_iter_" * string(k) * ".png")
end

function quadratic_penalty_main()
    x = [-1.0, 1.0]
    s = [0.0]   # only one slack
    ρ = 1e8     # large to force instability
    εa, εr = 1e-6, 1e-6
    εa_k, εr_k = 1e-2, 1e-2
    max_iter = 20

    residuals_x, residuals_c, residuals_g = Float64[], Float64[], Float64[]
    res0 = zeros(4)

    for k in 1:max_iter
        λ = -1/ρ * c_eq(x)
        μ =  1/ρ * (g_ineq(x) - s)
        rx = grad_f(x) - J_c(x)' * λ - J_g(x)' * μ
        rc = c_eq(x)
        rg = g_ineq(x) - s
        res = vcat(rx, rc, rg)
        println("[Iter \$k] ||res|| = ", norm(res), " | x = ", x, " | s = ", s)

        push!(residuals_x, norm(rx))
        push!(residuals_c, norm(rc))
        push!(residuals_g, norm(rg))
        res0 = k == 1 ? copy(res) : res0

        if norm(res) ≤ εa + εr * norm(res0)
            println("Converged at iteration \$k")
            break
        end

        x, s, path = newton_optimizer(x, s, ρ)
        Z = plot_penalty_contour(x, ρ, k)
        plot_bfgs_path_on_penalty(ρ, k, path, Z)
    end

    plot(1:length(residuals_x), residuals_x, lw=2, label="Stationarity", xlabel="Iteration", ylabel="Residual Norm", legend=:topright)
    plot!(1:length(residuals_c), residuals_c, lw=2, label="Equality Constraint")
    plot!(1:length(residuals_g), residuals_g, lw=2, label="Inequality Constraint")
    savefig("residuals_evolution.png")
    return x, s
end

x_sol, s_sol = quadratic_penalty_main()
