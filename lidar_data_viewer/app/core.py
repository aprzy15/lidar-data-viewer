from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vuetify, html
from trame.app import get_server
from pyvista.trame import PyVistaRemoteView
import pyvista
from PIL import Image
import numpy as np
import logging
import os
import io
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------
# TODO generate other data sources
# TODO add loading screen

class Engine:
    def __init__(self, server=None):
        if server is None:
            server = get_server()
        self._server = server
        # initialize state + controller
        state, ctrl = server.state, server.controller
        # Init variables
        self.active_town_file = {}
        self.feature_edge_actors = []
        self.boundary_mesh_fpaths = []
        self.town_info_dicts = []
        self.boundary_actors = []
        self.points_actors = None
        self.active_mesh_actor = None
        self.active_mesh_obj = None
        self.aspect_ratio = 0
        self.max_image_dim = 2000
        self.scale = 20000
        state.land_height_mult = 5
        state.water_emboss_layers = -10
        state.road_emboss_layers = 10
        state.building_height_mult = 1
        state.clip_border = True
        state.trame__title = "Lidar Viewer"

        self.colormaps = [
            {"text": "Terrain", "value": 'terrain'},
            {"text": "Cool Warm", "value": 'coolwarm'},
            {"text": "Viridis", "value": 'viridis'},
            {"text": "Jet", "value": 'jet'},
        ]
        self.ui_array_names = [
            {"text": "Elevation", "value": 'elevation'},
            {"text": "Boundary Mask", "value": 'boundary_mask'},
            {"text": "Building Heights", "value": 'building_height'},
            {"text": "Land Height", "value": 'land_map'},
            {"text": "Road Mask", "value": 'road_mask'},
            {"text": "Water Mask", "value": 'water_mask'},
        ]
        # Bind instance methods to state change
        self.bind_state_changes()
        # Load data
        self.init_data_source()
        self.plot_window = pyvista.Plotter()
        self.plot_window.set_background('black')
        self.load_boundary_meshes()
        self.update_active_mesh()
        ctrl.on_server_reload = self.ui
        state.setdefault("active_ui", None)
        state.active_ui = 'default'
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

    def init_data_source(self):
        current_directory = os.path.abspath(os.path.dirname(__file__))
        parent_folder = os.path.dirname(os.path.dirname(current_directory))
        data_folder = os.path.join(parent_folder, "data")
        files = os.listdir(data_folder)
        for file in files:
            ext = file.split('.')[-1]
            if ext == 'vtp':
                self.boundary_mesh_fpaths.append(os.path.join(data_folder, file))
            if ext == 'vti':
                town_name = file.split('_')[0]
                d = {"text": town_name, "value": os.path.join(data_folder, file)}
                self.town_info_dicts.append(d)
                if town_name == 'Somerville':
                    self.state.ui_town_dict = self.town_info_dicts[-1]
                    self.state.image_mesh_file = self.state.ui_town_dict['value']

    def bind_state_changes(self):
        self.state.change("image_width")(self.update_image_width)
        self.state.change("image_height")(self.update_image_height)
        self.state.change("active_cmap")(self.update_active_cmap)
        self.state.change("active_array_name")(self.update_active_array_name)
        self.state.change("apply_button")(self.update_active_mesh)
        self.state.change("show_labels")(self.update_show_labels)
        self.state.change("show_boundaries")(self.update_show_boundaries)
        self.ctrl.trigger('download_data')(self.download_data)

    def get_mesh_obj(self):
        if self.state.image_mesh_file != self.active_town_file:
            mesh = pyvista.read(self.state.image_mesh_file)
            self.state.image_height, self.state.image_width, _ = mesh.dimensions
            self.aspect_ratio = self.state.image_height / self.state.image_width
            self.active_mesh_obj = mesh
            self.active_town_file = self.state.image_mesh_file
            return self.active_mesh_obj
        else:
            self.state.image_width = self.state.image_width
            return self.active_mesh_obj

    def update_active_mesh(self, **kwargs):
        # Get vtk mesh object
        active_mesh_obj = self.get_mesh_obj()
        # Rescale display mesh if dimensions are too large
        self.state.disp_image_width, self.state.disp_image_height = self.get_disp_image_dim()
        grid = pyvista.create_grid(active_mesh_obj.copy(), dimensions=(self.state.disp_image_height, self.state.disp_image_width, 1))
        active_mesh_obj = grid.sample(active_mesh_obj.copy())
        # Warp mesh in 3D
        try:
            land_warp = active_mesh_obj.warp_by_scalar('land_map', factor=float(self.state.land_height_mult))
            water_warp = land_warp.warp_by_scalar('water_mask', factor=float(self.state.water_emboss_layers))
            road_warp = water_warp.warp_by_scalar('road_mask', factor=float(self.state.road_emboss_layers))
            display_mesh = road_warp.warp_by_scalar('building_height', factor=1 * float(self.state.building_height_mult))
        except ValueError:
            return
        if self.state.clip_border:
            display_mesh = display_mesh.threshold(0.5, scalars='boundary_mask')
        if float(self.state.water_emboss_layers) < 0:
            transform_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, float(self.state.water_emboss_layers) * -1],
                [0, 0, 0, 1],
            ])
            display_mesh = display_mesh.transform(transform_matrix)
        pd = display_mesh.point_data
        display_mesh.point_data['elevation'] = pd['land_map'] + pd['building_height'] + pd['road_mask']
        # Update plot window with new mesh
        if self.active_mesh_actor is not None:
            self.plot_window.remove_actor(self.active_mesh_actor)
        self.active_mesh_actor = self.plot_window.add_mesh(
            display_mesh,
            show_edges=False,
            scalars=self.state.active_array_name,
            cmap=self.state.active_cmap
        )
        self.plot_window.remove_scalar_bar()

    def get_disp_image_dim(self):
        try:
            width = int(self.state.image_width)
            height = int(self.state.image_height)
        except ValueError as err:
            return
        if (width > self.max_image_dim) or (height > self.max_image_dim):
            if width > height:
                width = 2000
                height = int(np.round(self.aspect_ratio * width))
            else:
                height = 2000
                width = int(np.round(height / self.aspect_ratio))
        return width, height

    def update_active_cmap(self, active_cmap, **kwargs):
        self.active_mesh_actor.mapper.SetLookupTable(pyvista.LookupTable(cmap=active_cmap))
        self.ctrl.view_update()

    def update_active_array_name(self, active_array_name, **kwargs):
        scalars = pyvista.get_array(self.active_mesh_actor.mapper.dataset, active_array_name, preference='point')
        self.active_mesh_actor.mapper.set_scalars(scalars, active_array_name, cmap=self.state.active_cmap)
        self.ctrl.view_update()

    def update_image_width(self, image_width, **kwargs):
        try:
            int(image_width)
        except ValueError as err:
            return
        image_width = int(image_width)
        if self.state.lock_aspect:
            self.state.image_width = image_width
            image_height = int(np.round(self.aspect_ratio * image_width))
            if image_height != self.state.image_height:
                self.state.image_height = image_height
        self.ctrl.view_update()

    def update_image_height(self, image_height, **kwargs):
        try:
            int(image_height)
        except ValueError as err:
            return
        image_height = int(image_height)
        if self.state.lock_aspect:
            self.state.image_height = image_height
            image_width = int(np.round(image_height / self.aspect_ratio))
            if image_width != self.state.image_width:
                self.state.image_width = image_width
            self.ctrl.view_update()

    def update_show_labels(self, show_labels, **kwargs):
        if show_labels:
            self.points_actors.VisibilityOn()
        else:
            self.points_actors.VisibilityOff()
        self.ctrl.view_update()

    def update_show_boundaries(self, show_boundaries, **kwargs):
        for boundary, edge in zip(self.boundary_actors, self.feature_edge_actors):
            if show_boundaries:
                boundary.VisibilityOn()
                edge.VisibilityOn()
            else:
                boundary.VisibilityOff()
                edge.VisibilityOff()
        self.ctrl.view_update()

    def load_boundary_meshes(self, **kwargs):
        centroids = []
        for fpath in self.boundary_mesh_fpaths:
            boundary_mesh = pyvista.read(fpath)
            self.boundary_actors.append(self.plot_window.add_mesh(boundary_mesh, show_edges=False, color='white'))
            feature_edges = boundary_mesh.extract_feature_edges(30)
            self.feature_edge_actors.append(self.plot_window.add_mesh(feature_edges, color="black", line_width=5))
            # Point labels
            offset = boundary_mesh.field_data.get('origin_offset')[0].split(',')
            offset = [float(val) for val in offset]
            centroid = boundary_mesh.field_data.get('centroid')[0].split(',')
            centroid = [float(val) for val in centroid]
            centroids.append([centroid[0]+offset[0], centroid[1]+offset[1], 0])
        point_labels = pyvista.PolyData(np.array(centroids))
        point_labels["labels"] = [d['text'] for d in self.town_info_dicts]
        self.points_actors = self.plot_window.add_point_labels(
            point_labels,
            "labels",
            point_size=10,
            font_size=12,
            shape_color='white',
            show_points=False
        )

    def download_data(self):
        print(f'downloading file {self.state.file_format}')
        # Resize mesh if need be
        dim = (int(self.state.image_height), int(self.state.image_width), 1)
        if dim != self.active_mesh_obj.dimensions:
            grid = pyvista.create_grid(self.active_mesh_obj.copy(), dimensions=dim)
            mesh = grid.sample(self.active_mesh_obj.copy())
        else:
            mesh = self.active_mesh_obj.copy()
            dim = (mesh.dimensions[0], mesh.dimensions[1])
        # Convert 1D arrays back to 2D and combine to single height map
        map_arr = (mesh['land_map'].reshape(dim, order='F')) * float(self.state.land_height_mult)
        map_arr += ((mesh['water_mask'].reshape(dim, order='F')) * float(self.state.water_emboss_layers))
        map_arr += ((mesh['road_mask'].reshape(dim, order='F')) * float(self.state.road_emboss_layers))
        map_arr += ((mesh['building_height'].reshape(dim, order='F')) * float(self.state.building_height_mult))
        map_arr -= map_arr.min()
        try:
            map_arr += int(self.state.num_base_layers)
        except ValueError:
            pass
        if self.state.clip_border:
            map_arr *= mesh['boundary_mask'].reshape(dim, order='F')
        map_arr = np.rot90(map_arr, k=1)
        # Rescale to uint16
        map_arr *= (65535 / map_arr.max())
        map_arr = np.round(map_arr).astype(np.uint16)
        map_arr = np.ascontiguousarray(map_arr)
        # Convert 2D array to byte array
        im = Image.fromarray(map_arr)
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
            # Top toolbar
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
                    vuetify.VSelect(
                        # Color Map
                        label="Map",
                        v_model=("image_mesh_file", self.state.ui_town_dict['value']),
                        items=("towns", self.town_info_dicts),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
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
                        html.Div("Display resolution: {{disp_image_height}}H {{disp_image_width}}W")
                        vuetify.VCheckbox(
                            label="Lock Aspect Ratio",
                            v_model=("lock_aspect", True),
                            dense=True,
                        )
                    vuetify.VTextField(
                        label="Base Layers",
                        v_model=("num_base_layers", 10),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Land height multiplier",
                        v_model=("land_height_mult", self.state.land_height_mult),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Water emboss layers",
                        v_model=("water_emboss_layers", self.state.water_emboss_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Road emboss layers",
                        v_model=("road_emboss_layers", self.state.water_emboss_layers),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VTextField(
                        label="Building height multiplier",
                        v_model=("building_height_mult", self.state.building_height_mult),
                        suffix=("currentSuffix", ""),
                    )
                    vuetify.VCheckbox(
                        label="Clip with border",
                        v_model=("clip_border", self.state.clip_border),
                        dense=True,
                    )
                    vuetify.VBtn(
                        "Apply",
                        outlined=False,
                        color='primary',
                        click=self.update_active_mesh,
                    )
                with ui_card(title="View Parameters"):
                    vuetify.VSelect(
                        # Color Map
                        label="Colormap",
                        v_model=("active_cmap", self.colormaps[0]['value']),
                        items=("colormaps", self.colormaps),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
                    vuetify.VSelect(
                        # Color Map
                        label="Color Array",
                        v_model=("active_array_name", self.ui_array_names[0]['value']),
                        items=("arr_names", self.ui_array_names),
                        hide_details=True,
                        dense=True,
                        outlined=True,
                        classes="pt-2",
                    )
                    vuetify.VCheckbox(
                        label="Show labels",
                        v_model=("show_labels", True),
                        dense=True,
                    )
                    vuetify.VCheckbox(
                        label="Show Boundaries",
                        v_model=("show_boundaries", True),
                        dense=True,
                    )
                with ui_card(title="Output Parameters"):
                    vuetify.VSelect(
                        # Color Map
                        label="Format",
                        v_model=("file_format", 'png'),
                        items=(
                            "file_formats",
                            {"text": "Height map (.png)", "value": 'png'},

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
                        click=f"utils.download('height_map.png', trigger('download_data'), 'image/png')",
                    )
            # Main content
            with layout.content:
                with vuetify.VContainer(
                        fluid=True,
                        classes="pa-0 fill-height",
                ):
                    view = PyVistaRemoteView(self.plot_window)
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera
                html.Div("image_height={{image_height}} image_width={{image_width}}")


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

