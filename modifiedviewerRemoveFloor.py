"""
OpenCV and Numpy Point cloud Software Renderer
This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.
Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

BG_DIST_TOLERANCE = 150
NUM_FRAMES_FOR_BG_AVG = 100
isPressed = 0

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)
bgFrameNumber = 0
backgroundDepthImage = np.zeros((240,320))
total = np.zeros((240,320))
while True:
    # Grab camera data
    tempDepth = np.zeros((240,320))
    print("frame number: ",bgFrameNumber)
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile( depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        tempDepth = depth_image
	

	#code for averaging the depth image
	
	if bgFrameNumber == NUM_FRAMES_FOR_BG_AVG:
		backgroundDepthImage = total / NUM_FRAMES_FOR_BG_AVG
	elif bgFrameNumber > 0 and bgFrameNumber < NUM_FRAMES_FOR_BG_AVG:
		total = depth_image + total
		bgFrameNumber = bgFrameNumber + 1
	average = total / bgFrameNumber
	#error = np.sum(average - depth_image)/76800
	#a = np.abs((average - depth_image)/depth_image)
	#a[a == np.inf] = np.nan
	#percentError = np.nanmean(a)
	#print("frame number: ",bgFrameNumber," percent error: ",percentError)

	
	#bgFrameNumber = bgFrameNumber + 1
	b = np.abs(average - depth_image)
	b[b == np.inf] = np.nan
	averageError = np.nanmean(b)
	#print("frame number: ",bgFrameNumber," avg error: ",averageError)
	#print("frame number: ",bgFrameNumber)

        color_image = np.asanyarray(color_frame.get_data())

        #BEGIN THE MODIFIED CODE

	originalColorImage = color_image
        #Convert BGR to HSV
        hsv  = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)

        #define range of pink color in HSV
        lower_pink = np.array([140,55,0])
        upper_pink = np.array([180,255,255])

        #mask has a 1 in the location if the pixel is in the pink range
        #and a 0 if the pixel is not in range
      	mask = cv2.inRange(hsv, lower_pink, upper_pink)

        #bitwise and between the mask and the color image will result in
        #all non pink pixels to become 0s
       	res = cv2.bitwise_and(color_image,color_image, mask= mask)
       	color_image = res
	color_image2 = color_image[::2,::2]
	
        #since the color image is twice the size of the depth image, we have to
        #take every other value
	mask2 = mask[::2,::2]
	color_image = cv2.bitwise_and(color_image2,color_image2,mask= mask2)
	

	for i in range(0,len(depth_image)):
            for j in range(0,len(depth_image[0])):

                #Test to see if this was supposed to be 0 or 100
                #print("color image[i][j][2]: ",color_image[i][j][2])
                #if color_image[i,j,2] > 0:
                    #Check if the value in backgroundDepthImage is near the value here
                    if isPressed:
                        #runs if backgroundDepthImage has non zero elements
                        #if maskedColorImage[i,j,0] > 0:
                        #print(depth_image[i,j]-backgroundDepthImage[i,j])
                        if depth_image[i,j] - backgroundDepthImage[i,j] >  BG_DIST_TOLERANCE:
                            #sets points that are not different from the background to white for visual purposes
                            color_image[i,j]=[255,255,255]
			    #color_image[2*i,2*j] = [255,255,255]
                            #color_image[2*i+1,2*j] = [255,255,255]
                            #color_image[2*i,2*j+1] = [255,255,255]
                            #color_image[2*i+1,2*j+1] = [255,255,255]
            #verts2 is a list of points, manualverts is a numpy array
        #default code
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)
        #end default code

    	verts2 = []
    	temp = []
    	intrin = depth_frame.profile.as_video_stream_profile().intrinsics
	
	#comment this out when done
	maskedColorImage = color_image
	color_image = originalColorImage

        #goes through every pixel. If the pixel in color was set to 0, don't add it
        #to verts2. In the end, verts2 will be a 2d array containing the (x,y,z)
        #values of the pink points
	
    	'''for i in range(0,len(depth_image)):
    	    for j in range(0,len(depth_image[0])):
        	
                #Test to see if this was supposed to be 0 or 100
		#print("color image[i][j][2]: ",color_image[i][j][2])
        	#if color_image[i,j,2] > 0:
                    #Check if the value in backgroundDepthImage is near the value here
                    if isPressed:
                        #runs if backgroundDepthImage has non zero elements
                        #if maskedColorImage[i,j,0] > 0:
			#print(depth_image[i,j]-backgroundDepthImage[i,j])
                        if depth_image[i,j] - backgroundDepthImage[i,j] <  BG_DIST_TOLERANCE:
                            a = rs.rs2_deproject_pixel_to_point(intrin, l, float(depth_image[i,j]))
                	    for elem in a:
                		temp.append(elem)

                	    verts2.append(temp)

    		    	    temp = []
			else:
			    
			    #sets points that are not different from the background to white for visual purposes
			    color_image[2*i,2*j] = [255,255,255]
			    color_image[2*i+1,2*j] = [255,255,255]
			    color_image[2*i,2*j+1] = [255,255,255]
			    color_image[2*i+1,2*j+1] = [255,255,255]
	    #verts2 is a list of points, manualverts is a numpy array
	'''
	manualverts = np.asanyarray(verts2)
	#print("verts shape: ", manualverts.shape)
	#np.savetxt("onlysphere.txt",verts)
	#print(verts.shape)
	color_image[:,:,:] = 255
        #using verts because we don't want to mess up the plotting
        #it plots with black for the ones we don't want even though the points
        #are still physically in their positions because the texture coords
        #use our masked color image

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key == ord("b"):
        print("B was pressed!")
        #code for getting the background
        bgFrameNumber = 1
	isPressed = 1

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break
    
    #if key == ord("k"):
	
# Stop streaming
pipeline.stop()
