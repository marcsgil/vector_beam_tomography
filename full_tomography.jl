using HDF5, QuantumMeasurements, CairoMakie, LinearAlgebra

includet("misc.jl")

function load_data(path, mode_name, symbols, sides)
    imgs = h5open(path) do file
        stack(
            read(file[mode_name*"/I$(pol)_$state"]) for pol ∈ symbols, state ∈ sides
        )
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
##
modes = Dict(
    "Hh" => kron([1.0, 0], [1, 0]),
    "LG+" => kron([1, im], [1, -im]) / 2,
    "LG-" => kron([1, -im], [1, +im]) / 2,
    "PHI+" => [1, 0, 0, 1] / √2
)

order = Dict(
    "Hh" => (true, true, true, true, true, true),
    "LG+" => (true, true, true, true, true, true),
    "LG-" => (false, true, false, true, true, true),
    "PHI+" => (true, true, false, true, false, true)
)

path = "Data/Batch1/cropped_data.h5"
##
mode_name = "LG+"
ϕ = modes[mode_name]
θ = traceless_vectorization(ϕ)

imgs = load_data(path, mode_name, symbols, sides);

μ = get_measurements(imgs, pol_states, order[mode_name])

sim = reshape(get_probabilities(μ, θ), size(imgs))

display(visualize(imgs, share_colorrange=true))
visualize(sim, share_colorrange=true)
##
for (mode, ϕ) ∈ modes
    imgs = load_data(path, mode, symbols, sides)

    μ = get_measurements(imgs, pol_states, order[mode_name])

    method = MaximumLikelihood()
    ϕ_pred = estimate_state(imgs, μ, method)[1] |> project2pure

    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(ϕ, ϕ_pred) * 100, digits=3), "%")
end
##
projector(ϕ) = ϕ * ϕ'
balanced_mix(args...) = sum(projector, args) / length(args)

structured_modes = Dict(
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
mode_name = "Hh"
ϕ = structured_modes[mode_name]
θ = traceless_vectorization(ϕ)

imgs = load_data(path, mode_name, symbols, sides);

μ = get_measurements(imgs, pol_states, order[mode_name])

sim = reshape(get_probabilities(μ, θ), size(imgs))



display(visualize(imgs))
visualize(sim)
##
for (mode, ϕ) ∈ structured_modes
    imgs_exp = load_data(path, mode, symbols, sides)

    μ = get_measurements(imgs_exp, pol_states, order[mode])

    method = MaximumLikelihood()
    ϕ_pred = estimate_state(imgs_exp, μ, method)[1]

    if ϕ isa AbstractVector
        ϕ_pred = project2pure(ϕ_pred)
    end

    imgs_theo = reshape(get_probabilities(μ, traceless_vectorization(ϕ)), size(imgs_exp))
    imgs_pred = reshape(get_probabilities(μ, traceless_vectorization(ϕ_pred)), size(imgs_exp))

    colorrange_exp = extrema(imgs_exp)
    colorrange_theo = extrema(imgs_theo)
    colorrange_pred = extrema(imgs_pred)

    fig = Figure(; figure_padding=0, size=(800, 950), title="Mode: $mode \n")

    g_theo = fig[1, 1] = GridLayout()
    g_exp = fig[2, 1] = GridLayout()
    g_pred = fig[3, 1] = GridLayout()


    for n ∈ axes(imgs_exp, 4), m ∈ axes(imgs_exp, 3)
        ax_theo = Axis(g_theo[n, m], aspect=DataAspect())
        ax_exp = Axis(g_exp[n, m], aspect=DataAspect())
        ax_pred = Axis(g_pred[n, m], aspect=DataAspect())

        for ax ∈ (ax_theo, ax_exp, ax_pred)
            hidedecorations!(ax)
        end

        heatmap!(ax_theo, imgs_theo[:, :, m, n], colormap=:jet, colorrange=colorrange_theo)
        heatmap!(ax_exp, imgs_exp[:, :, m, n], colormap=:jet, colorrange=colorrange_exp)
        heatmap!(ax_pred, imgs_pred[:, :, m, n], colormap=:jet, colorrange=colorrange_pred)
    end

    Label(g_theo[0, 1:6], "Theory (Mode: $mode)", fontsize=28)
    Label(g_exp[0, 1:6], "Experiment", fontsize=28)
    Label(g_pred[0, 1:6], "Reconstructed (Fidelity: $(round(fidelity(ϕ, ϕ_pred) * 100, digits=1))%)", fontsize=28)

    for g ∈ (g_theo, g_exp, g_pred)
        colgap!(g, 0)
        rowgap!(g, 0)
    end

    save("Results/$(mode).png", fig)
end