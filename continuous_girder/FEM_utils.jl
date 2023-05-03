using LinearAlgebra
using Plots
"""
    bernoulli_beam_2d(
        ex::AbstractArray{Float64,1},
        ey::AbstractArray{Float64,1},
        EI::Float64
    ) -> Ke

Local stiffness matrix for a 2D Bernoulli-Navier beam element.

TODO:
* maybe `Ke` should be a symmetric type
"""
function bernoulli_beam_2d(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    hinge::Int
)
    L = norm([diff(ex), diff(ey)])

    if hinge == 0 # no hinge
        Ke = EI / L * [
            12 / L^2      6 / L     -12 / L^2     6 / L;
            6 / L         4         -6 / L        2;
            -12 / L^2     -6 / L    12 / L^2      -6 / L;
            6 / L         2         -6 / L        4;
        ]'
    elseif hinge == 1 # right hinge
        Ke = EI / L * [
            3 / L^2       3 / L     -3 / L^2      0;
            3 / L         3         -3 / L        0;
            -3 / L^2      -3 / L    3 / L^2       0;
            0             0         0             0;
        ]'
    elseif hinge == -1 # left hinge
        Ke = EI / L * [
            3 / L^2       0         3 / L         -3 / L^2;
            0             0         0             0;
            3 / L         0         3             -3 / L;
            -3 / L^2      0         -3 / L        3 / L^2;
        ]'
    else
        throw("Unknown hinge value: $hinge (It should be 0, 1, or -1.)")
    end
    return Ke
end


"""
    bernoulli_beam_2d_moment(ex, ey, EI) -> Me

Bending moment at the ends of a 2D Bernoulli-Navier beam element.
"""
function bernoulli_beam_2d_moment(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    hinge::Int
)
    L = norm([diff(ex), diff(ey)])

    if hinge == 0 # no hinge
        Me = 2 * EI / L * [
            3 / L      -3 / L;
            2          -1;
            -3 / L     3 / L;
            1          -2;
        ]'
    elseif hinge == 1 # right hinge
        Me = 3 * EI / L * [
            1 / L      0;
            1          0;
            -1 / L     0;
            0          0;
        ]'
    elseif hinge == -1 # left hinge
        Me = 3 * EI / L * [
            0      -1 / L;
            0      0;
            0      1 / L;
            0      -1;
        ]'
    else
        throw("Unknown hinge value: $hinge (It should be 0, 1, or -1.)")
    end
    return Me
end



"""
    bernoulli_beam_2d(
        ex::AbstractArray{Float64,1},
        ey::AbstractArray{Float64,1},
        EI::Float64
    ) -> Ke

Local stiffness matrix for a 2D Bernoulli-Navier beam element.

TODO:
* maybe `Ke` should be a symmetric type
"""
function bernoulli_beam_2d_torsion(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    GJ::Float64,
    hinge::Int
)
    L = norm([diff(ex), diff(ey)])
    Ct = GJ / EI

    if hinge == 0 # no hinge
        Ke = EI / L * [
            12 / L^2      6 / L     0       -12 / L^2     6 / L    0;
            6 / L         4         0       -6 / L        2        0;
            0             0         Ct        0           0       -Ct;
            -12 / L^2     -6 / L    0        12 / L^2    -6 / L   0;
            6 / L         2         0        -6 / L       4        0;
            0             0        -Ct        0           0        Ct;
        ]'
        
    elseif hinge == 1 # right hinge
        Ke = EI / L * [
            3 / L^2       3 / L     0        -3 / L^2     0        0;
            3 / L         3         0        -3 / L       0        0;
            0             0         Ct        0           0       -Ct;
           -3 / L^2      -3 / L     0         3 / L^2     0        0;
            0             0         0         0           0        0;
            0             0        -Ct        0           0        Ct;
        ]'

    elseif hinge == -1 # left hinge
        Ke = EI / L * [
            3 / L^2       0         0       3 / L      -3 / L^2     0;
            0             0         0       0           0           0;
            0             0         Ct      0           0          -Ct;
            3 / L         0         0       3          -3 / L       0;
            -3 / L^2      0         0      -3 / L       3 / L^2     0;
            0             0        -Ct      0           0           Ct;
        ]'

    else
        throw("Unknown hinge value: $hinge (It should be 0, 1, or -1.)")
    end
    return Ke
end


"""
spring_element_4dof(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    L::Float64,
    ) -> Ke

Local stiffness matrix of a combined rotational and vertical spring
element, transformed to account for rigid links. This type of element
is used to connect the two girders of the bridge.

The order of dofs considered is: v, θ, ϕ, where θ is a bending dofs
and ϕ is a torsional dof.
"""
function spring_element_4dof(
    ey::AbstractArray{Float64,1},
    Kv::Float64,
    Kphi::Float64,
)
    e_ij = norm(diff(ey))
    Kc = e_ij^2 * Kv + Kphi

    # Element stiffness matrix referred to the master nodes, with one
    # vertical translation and one torsional dof per node.
    Ke = [
        Kv            e_ij*Kv     -Kv            e_ij*Kv;
        e_ij*Kv       Kc + Kphi    -e_ij*Kv      Kc - Kphi;
        -Kv          -e_ij*Kv      Kv            -e_ij*Kv;
        e_ij*Kv     Kc - Kphi     -e_ij*Kv       Kc + Kphi;
    ]'

    return Ke
end


function spring_element_4dof_test()

    f = [-1.0; -0.0; -0.2; 0.0] .* 1.0
    support_dofs = [0 0; 0 0]
    support_stiff = [1.0 1.0; 1.0 1.0]
    ey = [0.0; 5.0]
    Kv = 10000.0
    Kphi = 1000000.0
    sf_rot = 1.0
    sf_disp = 1.0

    e_ij = norm(diff(ey))
    Kc = e_ij^2 * Kv

    # Support dofs
    idx_remove = []
    for ii in 1:size(support_dofs)[1]
        support_dof = support_dofs[ii, :]
        if any(support_dof .!= 0)
            if support_dof[1] == 1
                append!(idx_remove, 2 * ii - 1)
            end
            if support_dof[2] == 1
                append!(idx_remove, 2 * ii)
            end
        end
    end

    # Drop the constrained dofs (rows and columns)
    _f(x) = x .!= (1:4)
    if isempty(idx_remove)
        idx_keep = 1:4
    else
        idx_keep = vec(prod(hcat(_f.(idx_remove)...), dims=2))
    end

    print("idx_keep = ")
    println(idx_keep)

    # Element stiffness matrix referred to the master nodes, with one
    # vertical translation and one torsional dof per node.
    Ke = [
        Kv            e_ij*Kv     -Kv            e_ij*Kv;
        e_ij*Kv       Kc + Kphi    -e_ij*Kv      Kc - Kphi;
        -Kv          -e_ij*Kv      Kv            -e_ij*Kv;
        e_ij*Kv     Kc - Kphi     -e_ij*Kv       Kc + Kphi;
    ]'

    # Transformation matrix used to obtain the stiffness matrix in 
    # the previous step
    Te = [1 e_ij 0 0; 0 1 0 0; 0 0 1 -e_ij; 0 0 0 1]
    fe = f #transpose(Te) * f
    print("transformed f = ")
    println(fe)

    # Keep free dofs
    Ke = Ke + diagm(vec(support_stiff')) 
    print("Kspring = ")
    println(diagm(vec(support_stiff')))

    print("K = ")
    println(Ke)

    Ks = Ke[idx_keep[:], idx_keep[:]]
    print("Ks = ")
    println(Ks)

    print("fs = ")
    println(f[idx_keep])

    # Solve
    us = Ks \ fe[idx_keep]
    u = zeros(4)
    u[idx_keep] = us

    print("f = ")
    println(f)

    println(support_dofs)

    print("u = ")
    println(u)

    # Plot
    z = 1.0
     _rmat(a) = [cos(a) -sin(a); sin(a) cos(a)]
    pts_coords = [0.0 0.0 0.0 0.0; -z z -z z;]
    theta = u[2:2:end]
    v = u[1:2:end]
    colors = ["red", "blue"]

    pl = plot()   
    for i in 1:2
        x_offset = [-e_ij/2 -e_ij/2 e_ij/2 e_ij/2]
        
        rot_mat = _rmat(theta[i] * sf_rot) * pts_coords[:, (2*i-1):2*i]
        rot_mat[2, :] .+= v[i] * sf_disp
        print(i)
        println(rot_mat[1, :])
        println(pts_coords[1, (2*i-1):2*i])
        plot!(x_offset[(2*i-1):2*i], pts_coords[2, (2*i-1):2*i], linestyle=:dash, linewidth=3, alpha=0.5, c=colors[i])
        plot!(x_offset[(2*i-1):2*i] + rot_mat[1, :], rot_mat[2, :], linewidth=5, c=colors[i])
    end
    display(pl)
    

end

# spring_element_4dof_test()