from packages import mesh2depth, pixel2mesh


def inundation_depth(mesh2obj_dec, front_view=None, ratio_height=None):
    pixel2mesh.pixel2obj()
    if mesh2obj_dec == 0:
        front_view, ratio_height = mesh2depth.obj2height(0)
        return front_view, ratio_height
    elif mesh2obj_dec == 1:
        inundation_depth = mesh2depth.obj2height(1, front_view, ratio_height)
        return inundation_depth
