using CairoMakie, GeometryBasics, HDF5, LinearAlgebra, QuantumMeasurements

mode_name = "PHI(+)+Vv"

ρ = h5open("Results/density_mats_projected.h5") do file
    read(file["$mode_name/prediction"])
end

with_theme(theme_latexfonts()) do
    fig = Figure(;fontsize=20)
    ax = Axis3(fig[1, 1],
        xticks=(1:4, [L"| H, H \rangle", L"| H, V \rangle", L"| V, H \rangle", L"| V, V \rangle"]),
        yticks=(1:4, [L"| H, H \rangle", L"| H, V \rangle", L"| V, H \rangle", L"| V, V \rangle"]),
        xlabelvisible=false,
        ylabelvisible=false,
        zlabel=L"\Re(\rho)",
    )

    data = real(ρ)
    meshscatter!(
        ax, axes(data, 1), axes(data, 2), fill(0, size(data)),
        marker=Rect3D(Vec3f0(-0.8, -0.8, 0), Vec3f0(1)),
        markersize=Vec3f0.(0.8, 0.8, vec(data)),
    )
    save("Results/mat_$mode_name.pdf", fig)
    fig
end