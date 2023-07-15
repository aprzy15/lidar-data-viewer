r"""
Define your classes and create the instances that you need to expose
"""
import logging
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout, SinglePageWithDrawerLayout
from trame.ui.router import RouterViewLayout
from trame.widgets import vuetify, vtk, html, router, trame, grid
from vtk_generation import get_vtk_obj, get_vtk_obj_pv
import numpy as np
import os
from PIL import Image
import io

from vtkmodules.vtkCommonCore import vtkLookupTable
from pyvista.trame import PyVistaRemoteView

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
import vtkmodules.vtkRenderingOpenGL2  # noqa
import pyvista
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Representation:
    Points = 0
    Wireframe = 1
    Surface = 2
    SurfaceWithEdges = 3


class LookupTable:
    terrain = 0
    coolwarm = 1
    viridis = 2
    jet = 3



# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------

class Engine:
    def __init__(self, server=None):
        if server is None:
            server = get_server()

        self._server = server

        # initialize state + controller
        state, ctrl = server.state, server.controller

        # Set state variable
        state.trame__title = "lidar-data-viewer"
        state.resolution = 20
        self.z_land_exaggeration = 5
        self.water_emboss_layers = -10
        self.road_emboss_layers = 10
        self.base_thickness = 20
        self.building_warp = 1
        self.clip_border = True
        self.lock_aspect = True
        self.actor = None
        self.colormaps = [
            {"text": "Terrain", "value": 'terrain'},
            {"text": "Cool Warm", "value": 'coolwarm'},
            {"text": "Viridis", "value": 'viridis'},
            {"text": "Jet", "value": 'jet'},
        ]
        self.active_cmap = self.colormaps[0]['value']
        self.ui_array_names = [
            {"text": "Elevation", "value": 'elevation'},
            {"text": "Boundary Mask", "value": 'boundary_mask'},
            {"text": "Building Heights", "value": 'building_height'},
            {"text": "Land Height", "value": 'land_map'},
            {"text": "Road Mask", "value": 'road_mask'},
            {"text": "Water Mask", "value": 'water_mask'},
        ]
        self.active_array = self.ui_array_names[0]['value']
        self.file_formats = [
            {"text": "Height map (.png)", "value": 'png'},
            {"text": "Mesh (.stl)", "value": 'stl'},
        ]
        self.active_file_format = self.file_formats[0]['value']


        state.change("image_width")(self.update_image_width)
        state.change("image_width")(self.update_image_width)

        state.change("file_format")(self.update_file_format)
        state.change("colormap")(self.update_colormap)
        state.change("plot_array")(self.update_plot_array)
        state.change("lock_aspect")(self.update_lock_aspect)
        state.change("z_land_exaggeration")(self.update_z_multiplier)
        state.change("water_emboss_layers")(self.update_water_emboss_layers)
        state.change("road_emboss_layers")(self.update_road_emboss_layers)
        state.change("base_thickness")(self.update_base_thickness)
        state.change("building_warp")(self.update_building_warp)
        state.change("clip_border")(self.update_clip_border)
        state.change("apply_button")(self.update_pipeline)
        ctrl.trigger('download_data')(self.download_data)


        # Load data
        self.plot_window = pyvista.Plotter()
        self.plot_window.set_background('black')


        CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIRECTORY)), r"data\Somerville_s22000_grid.vti")
        self.inp_mesh = pyvista.read(filepath)
        self.state.image_height, self.state.image_width, _ = self.inp_mesh.dimensions
        self.aspect_ratio = self.state.image_height / self.state.image_width
        self.update_pipeline()
        # self.plot_window.remove_scalar_bar()
        # self.plot_window.enable_mesh_picking(callback=self.click_mesh, show=True, left_clicking=True, style='surface')
        # Bind instance methods to controller
        # ctrl.reset_resolution = self.reset_resolution
        ctrl.on_server_reload = self.ui

        # Bind instance methods to state change
        state.setdefault("active_ui", None)
        state.active_ui = 'default'

        # TODO show low poly boundary outline meshes for different towns at once
        # TODO Cast map texture to boundary mesh
        # TODO update vtk processing to start with two vti files. one for interior boundary and one for exterior boundary,
        #  warp both by scalars multiple times to get final mesh
        # TODO Dimensions options

        # TODO show axis grid option
        # TODO model view parameters (color map, mesh style)

        # TODO export as STL
        # TODO export as image stack

        # Generate UI
        self.ui()
        self.ctrl.view_update()


    @property
    def server(self):
        return self._server

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller


    def show_in_jupyter(self, **kwargs):
        from trame.app import jupyter

        logger.setLevel(logging.WARNING)
        jupyter.show(self.server, **kwargs)

    def update_file_format(self, file_format, **kwargs):
        self.active_file_format = file_format

    def update_colormap(self, colormap, **kwargs):
        self.active_cmap = colormap
        self.actor.mapper.SetLookupTable(pyvista.LookupTable(cmap=colormap))
        self.ctrl.view_update()
        print('update cmap')

    def update_plot_array(self, plot_array, **kwargs):
        print(f'update color array {plot_array}')
        # self.actor.mapper.array_name = arr_name
        scalars = pyvista.get_array(self.actor.mapper.dataset, plot_array, preference='point')
        self.actor.mapper.set_scalars(scalars, plot_array, cmap=self.active_cmap)
        self.ctrl.view_update()
        # self.actor.mapper.set_scalars(arr_name)

    def update_image_width(self, image_width, **kwargs):
        print('update width')
        try:
            int(image_width)
        except ValueError as err:
            return
        image_width = int(image_width)
        if self.lock_aspect:
            self.state.image_width = image_width
            image_height = int(np.round(self.aspect_ratio * image_width))
            if image_width != self.state['image_height']:
                self.state['image_height'] = image_height
            self.ctrl.view_update()

    def update_image_height(self, image_height, **kwargs):
        print('update height')
        try:
            int(image_height)
        except ValueError as err:
            return
        image_height = int(image_height)
        if self.lock_aspect:
            self.state.image_height = image_height
            image_width = int(np.round(image_height / self.aspect_ratio))
            if image_width != self.state['image_width']:
                self.state['image_width'] = image_width
            self.ctrl.view_update()

    def update_lock_aspect(self, lock_aspect, **kwargs):
        try:
            self.lock_aspect = bool(lock_aspect)
        except ValueError as err:
            return

    def update_z_multiplier(self, z_land_exaggeration, **kwargs):
        print('-----------------------------------')
        try:
            self.z_land_exaggeration = float(z_land_exaggeration)
        except ValueError as err:
            pass

    def update_water_emboss_layers(self, water_emboss_layers, **kwargs):
        try:
            self.water_emboss_layers = float(water_emboss_layers)
        except ValueError as err:
            pass

    def update_road_emboss_layers(self, road_emboss_layers, **kwargs):
        try:
            self.road_emboss_layers = float(road_emboss_layers)
        except ValueError as err:
            pass

    def update_base_thickness(self, base_thickness, **kwargs):
        try:
            self.base_thickness = float(base_thickness)
        except ValueError as err:
            pass

    def update_building_warp(self, building_warp, **kwargs):
        try:
            self.building_warp = float(building_warp)
        except ValueError as err:
            return

    def update_clip_border(self, clip_border, **kwargs):
        try:
            self.clip_border = bool(clip_border)
        except ValueError as err:
            return

    def get_disp_image_dim(self):
        max_size = 2000
        try:
            int(self.state['image_width'])
        except ValueError as err:
            return
        try:
            int(self.state['image_height'])
        except ValueError as err:
            return
        width = int(self.state['image_width'])
        height = int(self.state['image_height'])
        if (width > max_size) or (height > max_size):
            if width > height:
                width = 2000
                height = int(np.round(self.aspect_ratio * width))
            else:
                height = 2000
                width = int(np.round(height / self.aspect_ratio))
        return width, height

    def update_pipeline(self, **kwargs):
        print(f'z_land_exaggeration: {self.z_land_exaggeration}')
        print(f'water_recess_layers: {self.water_emboss_layers}')
        print(f'road_emboss_layers: {self.road_emboss_layers}')
        print(f'base_thickness: {self.base_thickness}')
        print(f'clip_border: {self.clip_border}')
        print(self.inp_mesh.spacing)
        print('-----------------------------------------------')

        self.state.ui_width, self.state.ui_height = self.get_disp_image_dim()
        # TODO add vi
        # TODO add bbox selector
        # TODO add readout for visualization resolution reduction
        grid = pyvista.create_grid(self.inp_mesh.copy(), dimensions=(self.state.ui_height, self.state.ui_width, 1))
        mesh = grid.sample(self.inp_mesh.copy())

        land_warp = mesh.warp_by_scalar('land_map', factor=self.z_land_exaggeration)
        water_warp = land_warp.warp_by_scalar('water_mask', factor=self.water_emboss_layers)
        road_warp = water_warp.warp_by_scalar('road_mask', factor=self.road_emboss_layers)
        final_mesh = road_warp.warp_by_scalar('building_height', factor=1 * self.building_warp)
        if self.clip_border:
            final_mesh = final_mesh.threshold(0.5, scalars='boundary_mask')

        final_mesh.point_data['elevation'] = final_mesh.point_data['land_map'] + final_mesh.point_data['building_height'] + final_mesh.point_data['road_mask']
        if self.actor is not None:
            self.plot_window.remove_actor(self.actor)
        self.actor = self.plot_window.add_mesh(final_mesh, show_edges=False, scalars=self.active_array, cmap=self.active_cmap)
        self.plot_window.remove_scalar_bar()

    def download_data(self):
        # TODO units selector
        file_format = self.active_file_format
        print(f'downloading file {file_format}')
        dim = (int(self.state['image_height']), int(self.state['image_width']), 1)
        if dim != self.inp_mesh.dimensions:
            grid = pyvista.create_grid(self.inp_mesh.copy(), dimensions=dim)
            mesh = grid.sample(self.inp_mesh.copy())
        else:
            mesh = self.inp_mesh.copy()
        dim = (mesh.dimensions[0] - 1, mesh.dimensions[1] - 1)

        d = {}
        for name in mesh.array_names:
            mesh.get_array(name)
            array = np.array(mesh.get_array(name)).reshape(dim, order='F')
            d[name] = array

        # from matplotlib import pyplot as plt
        # import matplotlib
        map_arr = (d['land_map']/100) * self.z_land_exaggeration
        map_arr += ((d['water_mask']/100) * self.water_emboss_layers)
        map_arr += ((d['road_mask']/100) * self.road_emboss_layers)
        map_arr += ((d['building_height']/100) * self.building_warp)
        map_arr -= (map_arr.min() - 1)
        if self.clip_border:
            map_arr *= d['boundary_mask']

        # Rescale to uint16
        f = 65535 / map_arr.max()
        map_arr *= f
        map_arr = np.round(map_arr).astype(np.uint16)

        arr = np.ascontiguousarray(map_arr)
        im = Image.fromarray(arr)
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='PNG')
        opt = self.server.protocol.addAttachment(img_byte_arr.getvalue())
        return opt


    # --------------------------------------
    # UI
    # --------------------------------------

    def ui(self, *args, **kwargs):
        with SinglePageWithDrawerLayout(self._server) as layout:
            layout.title.set_text("Lidar Viewer")
            trame.LifeCycleMonitor(name="Root", events="['created', 'destroyed']")
            with layout.toolbar:
                vuetify.VSpacer()
                vuetify.VDivider(vertical=True, classes="mx-2")
                vuetify.VCheckbox(
                    v_model="$vuetify.theme.dark",
                    on_icon="mdi-lightbulb-off-outline",
                    off_icon="mdi-lightbulb-outline",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                )
                vuetify.VCheckbox(
                    v_model=("viewMode", "local"),
                    on_icon="mdi-lan-disconnect",
                    off_icon="mdi-lan-connect",
                    true_value="local",
                    false_value="remote",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                )
                with vuetify.VBtn(icon=True, click=self.plot_window.reset_camera):
                    vuetify.VIcon("mdi-crop-free")

            # Side screen drawer
            with layout.drawer as drawer:
                drawer.width = 325
                with ui_card(title="Model Parameters"):
                    with vuetify.VRow(classes="pt-2", dense=True):
                        with vuetify.VCol(cols="6"):
                            vuetify.VTextField(
                                label="Image Width",
                                v_model=("image_width", 0),
                                # value="get('image_width')",
                                suffix=("currentSuffix", ""),
                            )
                        with vuetify.VCol(cols="6"):
                            vuetify.VTextField(
                                label="Image Height",
                                v_model=("image_height", 0),
                                # value="get('image_height')",
                                suffix=("currentSuffix", ""),
                            )
                        html.Div("Display resolution: {{ui_height}}H {{ui_width}}W")
                        vuetify.VCheckbox(
                            label="Lock Aspect Ratio",
                            v_model=("lock_aspect", self.lock_aspect),
                            dense=True,
                        )
                    vuetify.VTextField(
                        label="Base Layers",
                        v_model=("z_land_exaggeration", self.z_land_exaggeration),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Land height multiplier",
                        v_model=("z_land_exaggeration", self.z_land_exaggeration),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Water emboss layers",
                        v_model=("water_emboss_layers", self.water_emboss_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Road emboss layers",
                        v_model=("road_emboss_layers", self.road_emboss_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Building height multiplier",
                        v_model=("building_warp", self.building_warp),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VCheckbox(
                        label="Clip with border",
                        v_model=("clip_border", self.clip_border),
                        dense=True,
                    )
                    vuetify.VBtn(
                        "Apply",
                        outlined=False,
                        color='primary',
                        click=self.update_pipeline,
                    )

                with ui_card(title="View Parameters"):
                    vuetify.VSelect(
                        # Color Map
                        label="Colormap",
                        v_model=("colormap", self.active_cmap),
                        items=(
                            "colormaps",
                            self.colormaps

                        ),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
                    vuetify.VSelect(
                        # Color Map
                        label="Color Array",
                        v_model=("plot_array", self.active_array),
                        items=(
                            "arr_names",
                            self.ui_array_names,
                        ),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
                with ui_card(title="Output Parameters"):
                    vuetify.VSelect(
                        # Color Map
                        label="Format",
                        v_model=("file_format", self.file_formats[0]),
                        items=(
                            "file_formats",
                            self.file_formats

                        ),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
                    vuetify.VBtn(
                        "Download",
                        classes="mt-3",
                        outlined=False,
                        left=True,
                        color='primary',
                        click="utils.download('height_map.png', trigger('download_data'), 'image/png')",
                    )

            # Main content
            with layout.content:
                # html.Div("image_height={{image_height}} image_width={{image_width}}")
                with vuetify.VContainer(
                        fluid=True,
                        classes="red lighten fill-height",
                ):
                    router.RouterView()

        with RouterViewLayout(self._server, "/", classes="blue lighten fill-height", fluid=True):
            print('LAYOUT')
            trame.LifeCycleMonitor(name="vtk", events="['created', 'destroyed']")

            view = PyVistaRemoteView(self.plot_window)
            view.update()
            self.ctrl.view_update.add(view.update)

            # self.ctrl.view_update = view.update
            # self.ctrl.view_reset_camera = view.reset_camera

        with RouterViewLayout(self._server, "/home"):
            trame.LifeCycleMonitor(name="Home", events="['created', 'destroyed']")
            with vuetify.VCard():
                vuetify.VCardTitle("This is home")
            vuetify.VBtn(
                "VTk",
                outlined=False,
                color='primary',
                # click=self.update_pipeline,
                to='/'
            )



# html.Div("image_height={{image_height}} image_width={{image_width}}")
def create_engine(server=None):
    # Get or create server
    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    return Engine(server)


def ui_card(title, ui_name='default'):
    with vuetify.VCard(v_show=f"active_ui == '{ui_name}'"):
        vuetify.VCardTitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
            hide_details=True,
            dense=True,
        )
        content = vuetify.VCardText(classes="py-2")
    return content

