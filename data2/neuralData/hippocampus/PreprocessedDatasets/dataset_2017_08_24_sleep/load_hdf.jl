using HDF5
function fromHDF5spikes()
    hf5 = h5open("2017-08-24_09-36-44.hdf5","r")
    spike_raster = []
    for (ind,(k,v)) in enumerate(pairs(read(hf5["ephys"])))
        push!(spike_raster,[])
        push!(spike_raster[ind],v["spikes"]["times"])
    end
    spike_raster
end
spike_raster = fromHDF5spikes()