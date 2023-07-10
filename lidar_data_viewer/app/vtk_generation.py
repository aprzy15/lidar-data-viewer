import numpy as np
# import cv2
import vtk.util.numpy_support as numpy_support
import vtk


def numpyToVTK(data, name):
    data_type = vtk.VTK_FLOAT
    shape = data.shape
    flat_data_array = data.flatten(order="F")
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    vtk_data.SetName(name)
    return vtk_data

def get_vtk_obj(
        data,
        z_land_exaggeration=5,
        water_recess_layers=10,
        road_emboss_layers=10,
        base_thickness=20,
        clip_border=False,
):
    d = data
    water_height = np.round(np.mean(d['land_map'] * d['water_mask']))
    land_map_stretched = d['land_map'] * z_land_exaggeration
    land_map_diff = land_map_stretched - d['land_map']
    d['building_map'] += land_map_diff
    d['road_mask'] += d['bridge_mask']
    d['road_mask'][np.where(d['road_mask'] > 1)] = 1
    land_map_new = (d['land_map'] + water_recess_layers) * ((d['water_mask'] * -1) + 1)
    full_map = land_map_new + (d['water_mask'] * water_height)
    full_map = full_map + (d['road_mask'] * road_emboss_layers)
    full_map = (full_map * ((d['building_mask'] * -1) + 1)) + (
            d['building_map'] + (d['building_mask'] * (road_emboss_layers + 10)))
    base_layer = np.zeros(full_map.shape)
    # Add Text
    # full_map = add_text(full_map)
    # Add base
    full_map += base_thickness
    # if clip_border:
    #     full_map *= d['boundary_mask']

    full_map = np.dstack([np.flipud(full_map), base_layer])

    # Create vtk image
    img = vtk.vtkImageData()
    img.SetDimensions(full_map.shape[0], full_map.shape[1], full_map.shape[2])
    # Add height point data
    img.GetPointData().AddArray(numpyToVTK(full_map, name='height'))
    # Add boundary mask
    b_mask = np.dstack([d['boundary_mask']]*2)
    img.GetPointData().AddArray(numpyToVTK(b_mask, name='boundary_mask'))

    alg = vtk.vtkWarpScalar()
    if clip_border:
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(img)
        threshold.ThresholdByUpper(0.5)
        threshold.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", 'boundary_mask')
        threshold.Update()
        alg.SetInputData(threshold.GetOutput())
    else:
        alg.SetInputData(img)


    # alg.SetInputData(alg.GetOutput())
    alg.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", 'height')
    alg.SetScaleFactor(1)
    alg.Update()

    return alg.GetOutput()
