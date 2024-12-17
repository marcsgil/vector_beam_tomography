using HDF5, QuantumMeasurements, LinearAlgebra

includet("misc.jl")

function load_data(path, mode_name, symbols, sides)
    imgs = h5open(path) do file
        stack(
            read(file[mode_name*"/I$(pol)_$state"]) for pol ∈ symbols, state ∈ sides
        )
    end

    for j ∈ 1:2:size(imgs, 3)
        normalize!(view(imgs, :, :, j:j+1, :), 1)
    end
    normalize!(imgs, 1)

    imgs
end

function get_measurements(imgs, pol_states, direct_first)
    xs = axes(imgs, 1)
    ys = axes(imgs, 2)
    rs = Iterators.product(xs, ys)
    sqrt_δA = 1 / sqrt(size(imgs, 3) * size(imgs, 4))

    num_outcomes = prod(size(imgs))
    μ = empty_measurement(num_outcomes, 4, Matrix{Float32})

    img_len = size(imgs, 1) * size(imgs, 2)

    counter = 0
    for n ∈ axes(imgs, 4), m ∈ axes(imgs, 3)
        counter += 1
        slice = view(imgs, :, :, m, n)
        pars = (sqrt_δA, center_of_mass_and_waist(slice, 1)...)

        phase = isodd(n) == direct_first[m] ? 0.0 : -π / 2

        idxs = img_len*(counter-1)+1:img_len*counter

        μ[idxs, :] = assemble_measurement_matrix(
            kron(fixed_order_basis(1, r, pars, phase), pol_states[m]) for r ∈ rs
        )
    end

    μ
end
##
symbols = [:H, :V, :D, :A, :R, :L]
pol_states = [polarization_state(s) for s in symbols]

sides = ["left", "right"]
projector(ϕ) = ϕ * ϕ'
balanced_mix(args...) = sum(projector, args) / length(args)

modes = Dict(
    "Hh" => kron([1.0, 0], [1, 0]),
    "Hh+Vv" => balanced_mix(kron([1, 0], [1, 0]), kron([0, 1], [0, 1])),
    "LG(-)" => kron([1, -im], [1, im]) / 2,
    "LG(-)+Vv" => balanced_mix(kron([1, -im], [1, im]) / 2, kron([0, 1], [0, 1])),
    "LG(+)" => kron([1, im], [1, -im]) / 2,
    "LG(+)+LG(-)" => balanced_mix(kron([1, im], [1, -im]) / 2, kron([1, -im], [1, im]) / 2),
    "LG(+)+Vv" => balanced_mix(kron([1, im], [1, -im]) / 2, kron([0, 1], [0, 1])),
    "PHI(+)" => [1, 0, 0, 1] / √2,
    "PHI(+)+Vv" => balanced_mix([1, 0, 0, 1] / √2, kron([0, 1], [0, 1])),
    "Vv" => kron([0, 1.0], [0, 1]),
)

order = Dict(
    "Hh" => (true, true, true, true, true, true),
    "Hh+Vv" => (true, true, true, true, true, true),
    "LG(-)" => (false, true, false, true, true, true),
    "LG(-)+Vv" => (false, true, false, true, true, true),
    "LG(+)" => (false, true, false, true, false, true),
    "LG(+)+LG(-)" => (false, true, false, true, false, true),
    "LG(+)+Vv" => (false, true, false, true, false, true),
    "PHI(+)" => (true, true, true, false, false, true),
    "PHI(+)+Vv" => (true, true, true, false, false, true),
    "Vv" => (true, true, true, true, true, true),
)

path = "Data/Batch2/cropped_data.h5"
##
for (mode, ρ) ∈ modes
    imgs_exp = load_data(path, mode, symbols, sides)

    μ = get_measurements(imgs_exp, pol_states, order[mode])

    method = MaximumLikelihood()
    σ = estimate_state(imgs_exp, μ, method)[1]

    """if ρ isa AbstractVector
        ψ = project2pure(σ)
        σ = ψ * ψ'
    end"""

    fid = fidelity(ρ, σ)

    h5open("Results/density_mats_not_projected.h5", "cw") do file
        file["$mode/true"] = ρ
        file["$mode/prediction"] = σ
        file["$mode/fidelity"] = fid
    end

    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fid * 100, digits=3), "%")
end