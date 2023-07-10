r"""
Define your classes and create the instances that you need to expose
"""
import logging
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import vuetify, vtk
from vtk_generation import get_vtk_obj
import numpy as np
import os
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
import vtkmodules.vtkRenderingOpenGL2  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Representation:
    Points = 0
    Wireframe = 1
    Surface = 2
    SurfaceWithEdges = 3


class LookupTable:
    Rainbow = 0
    Inverted_Rainbow = 1
    Greyscale = 2
    Inverted_Greyscale = 3


dataset_arrays = ['item1', 'item2', 'item3']
contour_value = 0.5
default_min = 1
default_max = 10
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
        self.water_recess_layers = 10
        self.road_emboss_layers = 10
        self.base_thickness = 20
        self.clip_border = True

        state.change("z_land_exaggeration")(self.update_z_multiplier)
        state.change("water_recess_layers")(self.update_water_recess_layers)
        state.change("road_emboss_layers")(self.update_road_emboss_layers)
        state.change("base_thickness")(self.update_base_thickness)
        state.change("clip_border")(self.update_clip_border)
        state.change("apply_button")(self.apply_model_params)

        # Load data
        CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIRECTORY)), r"data\Somerville_s22000.npy")
        self.data = {}
        with open(filepath, 'rb') as f:
            keys = np.load(f)
            for key in keys:
                self.data[key] = np.load(f)

        self.apply_model_params()
        self.init_render_window()

        # Bind instance methods to controller
        ctrl.reset_resolution = self.reset_resolution
        ctrl.on_server_reload = self.ui

        # Bind instance methods to state change
        state.setdefault("active_ui", None)
        state.active_ui = 'default'

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

    def update_z_multiplier(self, z_land_exaggeration, **kwargs):
        self.z_land_exaggeration = z_land_exaggeration

    def update_water_recess_layers(self, water_recess_layers, **kwargs):
        self.water_recess_layers = water_recess_layers

    def update_road_emboss_layers(self, road_emboss_layers, **kwargs):
        self.road_emboss_layers = road_emboss_layers

    def update_base_thickness(self, base_thickness, **kwargs):
        self.base_thickness = base_thickness

    def update_clip_border(self, clip_border, **kwargs):
        self.clip_border = clip_border

    def apply_model_params(self, **kwargs):
        print(f'z_land_exaggeration: {self.z_land_exaggeration}')
        print(f'water_recess_layers: {self.water_recess_layers}')
        print(f'road_emboss_layers: {self.road_emboss_layers}')
        print(f'base_thickness: {self.base_thickness}')
        print(f'clip_border: {self.clip_border}')
        print('-----------------------------------------------')
        self.vtk_obj = get_vtk_obj(
            self.data,
            z_land_exaggeration=self.z_land_exaggeration,
            water_recess_layers=self.water_recess_layers,
            road_emboss_layers=self.road_emboss_layers,
            base_thickness=self.base_thickness,
            clip_border=self.clip_border,
        )

    def init_render_window(self):
        renderer = vtkRenderer()
        self.renderWindow = vtkRenderWindow()
        self.renderWindow.AddRenderer(renderer)

        renderWindowInteractor = vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(self.renderWindow)
        renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # Mapper
        mesh_mapper = vtkDataSetMapper()
        mesh_mapper.SetInputData(self.vtk_obj)
        mesh_actor = vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        renderer.AddActor(mesh_actor)

        # LUT
        lut = vtkLookupTable()
        mesh_mapper.SetLookupTable(lut)

        lut.SetNumberOfColors(256)
        lut.SetHueRange(1.0, 0.0)
        lut.Build()

        scalar_range = self.vtk_obj.GetPointData().GetArray(0).GetRange()
        mesh_mapper.SelectColorArray('height')
        mesh_mapper.GetLookupTable().SetRange(scalar_range[0], scalar_range[1])
        mesh_mapper.SetScalarModeToUsePointFieldData()
        mesh_mapper.SetScalarVisibility(True)
        mesh_mapper.SetUseLookupTableScalarRange(True)

        # Mesh: Setup default representation to surface
        mesh_actor.GetProperty().SetRepresentationToSurface()
        mesh_actor.GetProperty().EdgeVisibilityOff()

        renderer.ResetCamera()


    def reset_resolution(self):
        self._server.state.resolution = 6

    # def on_resolution_change(self, resolution, **kwargs):
    #     logger.info(f">>> ENGINE(a): Slider updating resolution to {resolution}")

    # --------------------------------------
    # UI
    # --------------------------------------

    def ui(self, *args, **kwargs):
        with SinglePageWithDrawerLayout(self._server) as layout:
            layout.title.set_text("Lidar Viewer")
            # Top toolbar
            with layout.toolbar:
                vuetify.VSpacer()
                vuetify.VDivider(vertical=True, classes="mx-2")

            # Side screen drawer
            with layout.drawer as drawer:
                drawer.width = 325
                with ui_card(title="Model Parameters"):
                    vuetify.VTextField(
                        label="Z Multiplier",
                        v_model=("z_land_exaggeration", self.z_land_exaggeration),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Water Recess",
                        v_model=("water_recess_layers", self.water_recess_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Road Emboss",
                        v_model=("road_emboss_layers", self.road_emboss_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Base Thickness",
                        v_model=("base_thickness", self.base_thickness),
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
                        click=self.apply_model_params,
                    )

            # Main content
            with layout.content:
                with vuetify.VContainer(
                        fluid=True,
                        classes="pa-0 fill-height",
                ):
                    view = vtk.VtkRemoteView(self.renderWindow, interactive_ratio=1)
                    # view = vtk.VtkLocalView(self.renderWindow)
                    # view = vtk.VtkRemoteLocalView(
                    #     renderWindow, namespace="view", mode="local", interactive_ratio=1
                    # )
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera


def create_engine(server=None):
    # Get or create server
    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    return Engine(server)

# UI elements ----------------------------------

def standard_buttons():
    # vuetify.VCheckbox(
    #     v_model=("cube_axes_visibility", True),
    #     on_icon="mdi-cube-outline",
    #     off_icon="mdi-cube-off-outline",
    #     classes="mx-1",
    #     hide_details=True,
    #     dense=True,
    # )
    vuetify.VCheckbox(
        v_model="$vuetify.theme.dark",
        on_icon="mdi-lightbulb-off-outline",
        off_icon="mdi-lightbulb-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model=("viewMode", "local"),  # VtkRemoteLocalView => {namespace}Mode=['local', 'remote']
        on_icon="mdi-lan-disconnect",
        off_icon="mdi-lan-connect",
        true_value="local",
        false_value="remote",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click="$refs.view.resetCamera()"):
        vuetify.VIcon("mdi-crop-free")

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


# def param_card():
#     with ui_card(title="Model Parameters"):
#         vuetify.VTextField(
#             label="Z Multiplier",
#             v_model=("z_land_exaggeration", 5),
#             suffix=("currentSuffix", ""),
#         )



