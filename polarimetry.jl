using QuantumMeasurements, LinearAlgebra, PrettyTables, HDF5

symbols = [:H, :V, :D, :A, :R, :L]
μ = assemble_measurement_matrix(polarization_state(s) for s in symbols)

file = h5open("Data/Batch2/cropped_data.h5")

"""modes = Dict(
    "Hh" => polarization_state(:H),
    "LG+" => polarization_state(:R),
    "LG-" => polarization_state(:L),
    "PHI+" => Matrix{Float32}(I, 2, 2) / 2
)"""

modes = Dict(
    "LG(+)" => polarization_state(:R),
    "LG(-)" => polarization_state(:L),
    "PHI(+)" => I(2) / 2,
    "Hh" => polarization_state(:H),
    "Vv" => polarization_state(:V),
    "Hh+Vv" => I(2) / 2
)

method = MaximumLikelihood()

for (mode, ϕ) ∈ modes
    p = [(sum(file[joinpath(mode, "I$(pol)_left")] |> read)
          +
          sum(file[joinpath(mode, "I$(pol)_right")] |> read)) for pol in symbols]

    normalize!(view(p, 1:2), 1)
    normalize!(view(p, 3:4), 1)
    normalize!(view(p, 5:6), 1)

    if ϕ isa Vector
        c = estimate_state(p, μ, method)[1] |> project2pure
    else
        c = estimate_state(p, μ, method)[1]
    end

    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(c, ϕ) * 100, digits=3), "%")
end
