#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from auxiliary.laserscan import LaserScan, SemLaserScan


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, gt_names, pred_names, offset=0, size=1, bgcolor='black',
               semantics=True, instances=False, gt_and_preds=False):
    self.scan = scan
    self.scan_names = scan_names
    if pred_names!=None and not gt_and_preds:
      self.gt_names = pred_names
      self.pred_names = None
    else:
      self.gt_names = gt_names
      self.pred_names = pred_names
    self.offset = offset
    self.size = size
    self.bgcolor = bgcolor
    if bgcolor=='white':
      self.bocolor = 'black'
    else:
      self.bocolor = 'white'
    self.show_diff=False
    self.total = len(self.scan_names)
    self.semantics = semantics
    self.instances = instances
    self.gt_and_preds = gt_and_preds

    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, plus, minus, difference, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor=self.bgcolor)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color=self.bocolor, parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    # add semantics
    if self.semantics and not (self.instances and self.gt_and_preds):
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      self.sem_view.camera.link(self.scan_view.camera)

    # add instances
    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2-self.gt_and_preds)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      self.inst_view.camera.link(self.scan_view.camera)

    if self.gt_and_preds:
      print("Using gt and preds in visualizer")
      self.preds_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.canvas.scene)
      self.grid.add_widget(self.preds_view, 0, 2)
      self.preds_vis = visuals.Markers()
      self.preds_view.camera = 'turntable'
      self.preds_view.add(self.preds_vis)
      visuals.XYZAxis(parent=self.preds_view.scene)
      self.preds_view.camera.link(self.scan_view.camera)


    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color=self.bocolor, parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics and not (self.instances and self.gt_and_preds):
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2-self.gt_and_preds, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)

    if self.gt_and_preds:
      self.preds_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.bocolor, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.preds_img_view, 2, 0)
      self.preds_img_vis = visuals.Image(cmap='viridis')
      self.preds_img_view.add(self.preds_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])
    if self.semantics:
      if self.gt_and_preds:
        self.scan.open_label(self.gt_names[self.offset], self.pred_names[self.offset])
      else:
        self.scan.open_label(self.gt_names[self.offset])
      self.scan.colorize(self.show_diff)

    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=self.size)

    # plot semantics
    if self.semantics and not (self.instances and self.gt_and_preds):
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=self.size)

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan.inst_label_color[..., ::-1],
                             edge_color=self.scan.inst_label_color[..., ::-1],
                             size=self.size)

    # plot predictions
    if self.gt_and_preds:
      self.preds_vis.set_data(self.scan.points,
                            face_color=self.scan.preds_label_color[..., ::-1],
                            edge_color=self.scan.preds_label_color[..., ::-1],
                            size=self.size)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if self.semantics and not (self.instances and self.gt_and_preds):
      self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
      self.sem_img_vis.update()

    if self.instances:
      self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()

    if self.gt_and_preds:
      self.preds_img_vis.set_data(self.scan.proj_preds_color[..., ::-1])
      self.preds_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'P':
      self.size += 1
      self.update_scan()
    elif event.key == 'M':
      self.size = max(1,self.size-1)
      self.update_scan()
    elif event.key == 'D':
      self.show_diff = not self.show_diff
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()


  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
