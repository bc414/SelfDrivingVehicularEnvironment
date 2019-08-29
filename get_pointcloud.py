import pyrealsense2 as rs
import numpy as np
pipe = rs.pipeline()
pipe.start()
frames = pipe.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()
pc = rs.pointcloud()
pc.map_to(color)
points = pc.calculate(depth)
vtx = np.asanyarray(points.get_vertices())
tex = np.asanyarray(points.get_texture_coordinates())
np.save("output.npy",vtx)
