from pyntcloud import PyntCloud
cloud = PyntCloud.from_file("sphere.ply")
# cloud.plot(use_as_color="z", point_size=2)

from pyntcloud.ransac import single_fit, RansacSphere
inliers, model = single_fit(cloud.xyz, RansacSphere, return_model=True)

print(model.center)
# array([  2.04683742e-15,   2.84217094e-14,  -5.06828830e-07])
print(cloud.xyz.mean(0))
# array([ -4.16906794e-08,   8.33813587e-08,   3.33525435e-07], dtype=float32)


print(model.radius)
# 24.999997417679129
print(cloud.xyz.ptp(0))
# array([ 50.        ,  49.94900131,  49.93859863], dtype=float32)
# ptp are the distances between min and max point along x, y and z axis
# makes sense to have radius = 25

