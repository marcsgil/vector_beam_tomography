using StatsBase, StructuredLight

function center_of_mass(img::AbstractMatrix{T}) where {T}
    m₀ = zero(T)
    n₀ = zero(T)
    for n ∈ axes(img, 2), m ∈ axes(img, 1)
        m₀ += m * img[m, n]
        n₀ += n * img[m, n]
    end
    m₀ / sum(img), n₀ / sum(img)
end

function center_of_mass_and_variance(img)
    T = float(typeof(firstindex(img)))

    m₀ = zero(T)
    n₀ = zero(T)
    s² = zero(T)
    N = zero(T)

    for n ∈ axes(img, 2), m ∈ axes(img, 1)
        m₀ += m * img[m, n]
        n₀ += n * img[m, n]
        s² += (m^2 + n^2) * img[m, n]
        N += img[m, n]
    end

    m₀ /= sum(img)
    n₀ /= sum(img)
    s² = s² / N - m₀^2 - n₀^2

    m₀, n₀, s²
end

function center_of_mass_and_waist(img, order)
    m₀, n₀, s² = center_of_mass_and_variance(img)
    m₀, n₀, √(2 * s² / (order + 1))
end

function fixed_order_basis!(dest, r, pars, phase=0)
    order = length(dest) - 1
    x, y = r
    sqrt_δA, x₀, y₀, w = pars
    for j ∈ eachindex(dest)
        m = order + 1 - j
        n = j - 1
        dest[j] = hg(x - x₀, y - y₀; w, m, n) * cis(-j * phase) * sqrt_δA
    end
    dest
end

function fixed_order_basis(order, r, pars, phase=0)
    buffer = Vector{complex(float(eltype(r)))}(undef, order + 1)
    fixed_order_basis!(buffer, r, pars, phase)
end

most_frequent_value(data) = countmap(data) |> argmax

function remove_background!(images, bg)
    map!(x -> x < bg ? zero(x) : x - bg, images, images)
end

function remove_background!(images)
    bg = most_frequent_value(images)
    remove_background!(images, bg)
end