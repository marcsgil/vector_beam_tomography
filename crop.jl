using ImageMagick, FileIO, StructuredLight, CairoMakie, HDF5
includet("misc.jl")

function format(img)
    reverse(img', dims=2)
end

function load_uint8(path)
    map(load(path)) do pixel
        reinterpret(UInt8, pixel.r)
    end |> format
end

function crop(img, target_size)
    X, Y = round.(Int, center_of_mass(img))
    δ = target_size ÷ 2

    if iseven(target_size)
        X_min = max(X - δ + 1, 1)
        Y_min = max(Y - δ + 1, 1)
    else
        X_min = max(X - δ, 1)
        Y_min = max(Y - δ, 1)
    end

    X_max = min(X + δ, size(img, 1))
    Y_max = min(Y + δ, size(img, 2))


    cropped = img[X_min:X_max, Y_min:Y_max]

    padded = zeros(float(eltype(img)), target_size, target_size)
    padded[1:size(cropped, 1), 1:size(cropped, 2)] = cropped

    padded
end

function crop_two_horizontal(img::Matrix, target_size)
    L = size(img, 1) ÷ 2
    crop(view(img, 1:L, :), target_size), crop(view(img, L+1:size(img, 1), :), target_size)
end
##
for root_path ∈ ("Data/Batch1", "Data/Batch2")
    saving_path = joinpath(root_path, "cropped_data.h5")

    for (root, dirs, files) in walkdir(root_path)
        mode_name = last(splitdir(root))

        for file ∈ files
            if occursin(".bmp", file)
                img = load_uint8(joinpath(root, file))
                remove_background!(img)
                left, right = crop_two_horizontal(img, 150)
                name = first(splitext(file))
                h5open(saving_path, "cw") do file
                    file[joinpath(mode_name, "$(name)_left")] = left
                    file[joinpath(mode_name, "$(name)_right")] = right
                end
            end
        end
    end
end