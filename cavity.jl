# Lid-driven cavity 2D — vorticidade–funcao-corrente (Re=100), versao VETORIZADA
# Uso: julia -O3 --check-bounds=no cavity.jl [N] [Re] [CFL] [tol] [JACOBI_ITERS]
# Ex.:  julia -O3 --check-bounds=no cavity.jl 256 100 0.3 1e-6 80

using Printf, Dates

function main()
    # ---------------- CLI ----------------
    N    = length(ARGS) >= 1 ? parse(Int,     ARGS[1]) : 256
    Re   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 100.0
    CFL  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.3
    tol  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-6
    JIT  = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 80

    L  = 1.0
    dx = L/(N-1);  dy = dx
    nu = 1.0/Re
    Utop = 1.0
    maxiter = 20_000

    # --------------- campos ---------------
    omega    = zeros(N, N)
    psi      = zeros(N, N)
    u        = zeros(N, N)
    v        = zeros(N, N)
    omeganew = similar(omega)
    psinew   = similar(psi)   # buffer para Jacobi vetorizado

    @views begin
        # ---------- helpers ----------
        set_psi_bcs!(psi) = (psi[:,1] .= 0.0; psi[:,end] .= 0.0; psi[1,:] .= 0.0; psi[end,:] .= 0.0)

        function update_uv_from_psi!(u, v, psi)
            u[2:end-1,2:end-1] .= (psi[2:end-1,3:end] .- psi[2:end-1,1:end-2]) ./ (2dy)
            v[2:end-1,2:end-1] .= -(psi[3:end,2:end-1] .- psi[1:end-2,2:end-1]) ./ (2dx)
            # paredes (no-slip; tampa U=1)
            u[:,1]   .= 0.0;  v[:,1]   .= 0.0
            u[:,end] .= Utop; v[:,end] .= 0.0
            u[1,:]   .= 0.0;  v[1,:]   .= 0.0
            u[end,:] .= 0.0;  v[end,:] .= 0.0
        end

        function update_wall_vorticity!(omega, psi)
            omega[:,1]   .= -2 .* psi[:,2]       ./ (dy^2)               # bottom
            omega[:,end] .= -2 .* psi[:,end-1]   ./ (dy^2) .- 2Utop/dy   # top (U=1)
            omega[1,:]   .= -2 .* psi[2,:]       ./ (dx^2)               # left
            omega[end,:] .= -2 .* psi[end-1,:]   ./ (dx^2)               # right
        end

        # omega-step (vetorizado)
        function step_vorticity!(omeganew, omega, u, v, dt)
            adv = u[2:end-1,2:end-1] .* (omega[3:end,2:end-1] .- omega[1:end-2,2:end-1]) ./ (2dx) .+
                  v[2:end-1,2:end-1] .* (omega[2:end-1,3:end] .- omega[2:end-1,1:end-2]) ./ (2dy)
            lap = (omega[3:end,2:end-1] .- 2 .* omega[2:end-1,2:end-1] .+ omega[1:end-2,2:end-1]) ./ (dx^2) .+
                  (omega[2:end-1,3:end] .- 2 .* omega[2:end-1,2:end-1] .+ omega[2:end-1,1:end-2]) ./ (dy^2)
            omeganew .= omega
            omeganew[2:end-1,2:end-1] .+= dt .* (-adv .+ nu .* lap)
        end

        # Poisson Jacobi (vetorizado com buffer)
        function jacobi_psi!(psi, omega, iters)
            dx2 = dx^2; dy2 = dy^2; beta = 1/(2*(dx2+dy2))
            for _ in 1:iters
                set_psi_bcs!(psi)
                psinew[2:end-1,2:end-1] .= beta .* (
                    (psi[3:end,2:end-1] .+ psi[1:end-2,2:end-1]) .* dy2 .+
                    (psi[2:end-1,3:end] .+ psi[2:end-1,1:end-2]) .* dx2 .+
                    (dx2*dy2) .* omega[2:end-1,2:end-1]
                )
                # manter fronteiras a zero no buffer
                psinew[:,1] .= 0.0; psinew[:,end] .= 0.0; psinew[1,:] .= 0.0; psinew[end,:] .= 0.0
                psi, psinew = psinew, psi   # swap
            end
            return psi
        end

        # ------------- loop temporal -------------
        t0 = time(); steps = 0; dt_acc = 0.0; res = Inf
        # warm-up leve
        set_psi_bcs!(psi); update_uv_from_psi!(u,v,psi); update_wall_vorticity!(omega,psi)
        step_vorticity!(omeganew, omega, u, v, 1e-6); psi = jacobi_psi!(psi, omega, 1)

        for it in 1:maxiter
            set_psi_bcs!(psi)
            update_uv_from_psi!(u, v, psi)
            update_wall_vorticity!(omega, psi)

            umax = max(maximum(abs.(u)), 1e-12)
            vmax = max(maximum(abs.(v)), 1e-12)
            dtc1 = dx/umax; dtc2 = dy/vmax
            dtd1 = dx^2/(4nu); dtd2 = dy^2/(4nu)
            dt = CFL * min(dtc1, dtc2, dtd1, dtd2)

            step_vorticity!(omeganew, omega, u, v, dt)

            # alinhar bordas antes do res (para nao contaminar)
            omeganew[1,:] .= omega[1,:];   omeganew[end,:] .= omega[end,:]
            omeganew[:,1] .= omega[:,1];   omeganew[:,end] .= omega[:,end]
            res = maximum(abs.(omeganew .- omega))

            omega .= omeganew
            psi = jacobi_psi!(psi, omega, JIT)

            steps += 1; dt_acc += dt
            if it % 200 == 0
                @printf("Iter %d, dt=%.2e, res=%.2e\n", it, dt, res)
                flush(stdout)
            end
            if res < tol
                @printf("Convergiu em %d passos\n", it)
                flush(stdout)
                break
            end
        end

        elapsed = time() - t0
        cells = (N-2)*(N-2)
        mlups_vort = cells*steps/elapsed/1e6
        avg_dt = dt_acc/max(steps,1)

        @printf("N=%d Re=%.1f steps=%d time=%.3fs MLUPS(vort)=%.2f avg_dt=%.2e res=%.2e\n",
                N, Re, steps, elapsed, mlups_vort, avg_dt, res)

        # ----------- saidas -----------
        data_dir = joinpath(pwd(), "data")
        isdir(data_dir) || mkpath(data_dir)

        open(joinpath(data_dir, "u_center_julia_N$(N).dat"), "w") do f
            for j in 1:N
                @printf(f, "%f %f\n", (j-1)*dy, u[cld(N,2), j])
            end
        end
        open(joinpath(data_dir, "v_center_julia_N$(N).dat"), "w") do f
            for i in 1:N
                @printf(f, "%f %f\n", (i-1)*dx, v[i, cld(N,2)])
            end
        end

        # CSV de performance continua na raiz
        sumfile = "summary_julia.csv"
        newfile = !isfile(sumfile)
        open(sumfile, "a") do f
            if newfile
                write(f, "timestamp,N,Re,CFL,tol,jacobi_iters,steps,time_s,avg_dt,MLUPS_vort,res\n")
            end
            @printf(f, "%s,%d,%.1f,%.3f,%.1e,%d,%d,%.6f,%.6e,%.3f,%.3e\n",
                    Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                    N, Re, CFL, tol, JIT, steps, elapsed, avg_dt, mlups_vort, res)
        end
    end # @views
end

main()
