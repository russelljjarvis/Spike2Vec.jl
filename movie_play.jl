wget https://drive.google.com/file/d/1--zhmjvQDF2UkoZRm6RDTp0zKrGed6em/view?usp=share_link
dataset = Odesa.NMNIST.NmnistMotion("./nmnist_motions.hdf5")

module NMNIST
using Printf
using HDF5
import HDF5
##
# For reference see this file:
##
# # https://github.com/yeshwanthravitheja/odesa_variants/blob/aecdc06976a26c593eff1d9a9b913840da52147a/convolutional/ODESA/Conv.py

mutable struct NmnistMotion
    file::HDF5.HDF5File

    function NmnistMotion(filename::String)
        dataset = h5open(filename, "r")
        new(dataset)
    end

end

function closeFile(dataset::NmnistMotion)
    close(dataset.file)
end

function get_training_count(dataset::NmnistMotion)
    return length(dataset.file["nmnist/train"])
end

function get_testing_count(dataset::NmnistMotion)
    return length(dataset.file["nmnist/test"])
end

function get_dataset_item(dataset::NmnistMotion, corpus::String, id::Int32)
    path = @sprintf "nmnist/%s/%05d" corpus id
    item = dataset.file[path]
    events::Matrix{Int32} = read(item)
    motion::Int32 = read(item["motion"])
    label::Int32 = read(item["label"])
    return (events, motion, label)
end

function get_element_dimensions()
    return (34, 34)
end

end # End of module
