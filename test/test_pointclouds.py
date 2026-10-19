import unittest
import numpy as np

import ros2_numpy as rnp
from sensor_msgs.msg import PointCloud2, PointField, Image

class TestPointClouds(unittest.TestCase):
    def test_writable_xyz_view(self):
        points = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        view = rnp.point_cloud2.get_xyz_points_view(points, readonly=False)
        view[1, 2] = 42
        self.assertEqual(points['z'][1], 42)
        del points
        self.assertEqual(view[1, 2], 42)

    def test_xyz_view(self):
        for dtype in ('f4', 'f8', '>f4'):
            points = np.zeros((3, 6), dtype=[
                ('intensity', 'u4'), ('x', dtype), ('y', dtype), ('z', dtype)])
            points['x'] = np.arange(18).reshape(3, 6)
            points['y'] = 2
            points['z'] = np.nan
            for source in (points, points[:, ::2], points[::-1, ::-1]):
                view = rnp.point_cloud2.get_xyz_points_view(source)
                self.assertTrue(np.shares_memory(view, source))
                self.assertEqual(view.dtype, np.dtype(dtype))
                self.assertFalse(view.flags.writeable)
                np.testing.assert_equal(view[..., 0], source['x'])
                np.testing.assert_equal(view[..., 1], source['y'])
                self.assertTrue(np.isnan(view[..., 2]).all())
            points['x'] = 42
            self.assertTrue((view[..., 0] == 42).all())

    def test_xyz_view_rejects_incompatible_fields(self):
        for dtype in ([('x', 'f4'), ('y', 'f8'), ('z', 'f4')],
                      [('x', 'f4'), ('intensity', 'f4'), ('y', 'f4'), ('z', 'f4')]):
            with self.assertRaises(ValueError):
                rnp.point_cloud2.get_xyz_points_view(np.zeros(3, dtype=dtype))

    def makeArray(self, npoints):
        points_arr = np.zeros((npoints,), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('r', np.uint8),
            ('g', np.uint8),
            ('b', np.uint8)])
        points_arr['x'] = np.random.random((npoints,))
        points_arr['y'] = np.random.random((npoints,))
        points_arr['z'] = np.random.random((npoints,))
        points_arr['r'] = np.floor(np.random.random((npoints,))*255)
        points_arr['g'] = 0
        points_arr['b'] = 255

        return points_arr

    def test_convert_dtype(self):
        fields = [
            PointField(name='x', offset=0, count=1,
                       datatype=PointField.FLOAT32),
            PointField(name='y', offset=4, count=1,
                       datatype=PointField.FLOAT32)
        ]
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32)
        ])
        conv_fields = rnp.msgify(PointField, dtype, plural=True)
        self.assertSequenceEqual(fields, conv_fields,
                                 'dtype->Pointfield Failed with simple values')

        conv_dtype = rnp.numpify(fields, point_step=8)
        self.assertSequenceEqual(dtype, conv_dtype,
                                 'dtype->Pointfield Failed with simple values')

    def test_convert_dtype_inner(self):
        fields = [
            PointField(name='x', offset=0, count=1,
                       datatype=PointField.FLOAT32),
            PointField(name='y', offset=4, count=1,
                       datatype=PointField.FLOAT32),
            PointField(name='vectors', offset=8, count=3,
                       datatype=PointField.FLOAT32)
        ]

        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('vectors', np.float32, (3,))
        ])

        conv_fields = rnp.msgify(PointField, dtype, plural=True)
        self.assertSequenceEqual(fields, conv_fields,
                                 'dtype->Pointfield with inner dimensions')

        conv_dtype = rnp.numpify(fields, point_step=8)
        self.assertEqual(dtype, conv_dtype,
                         'Pointfield->dtype with inner dimensions')


    def test_roundtrip(self):

        points_arr = self.makeArray(100)
        cloud_msg = rnp.msgify(PointCloud2, points_arr)
        new_points_arr = rnp.numpify(cloud_msg)

        np.testing.assert_equal(points_arr, new_points_arr)

    def test_roundtrip_numpy(self):
        points_arr = self.makeArray(100)
        cloud_msg = rnp.msgify(PointCloud2, points_arr)
        new_points_arr = rnp.numpify(cloud_msg)

        np.testing.assert_equal(points_arr, new_points_arr)

    def test_roundtrip_zero_points(self):
        """Test to make sure zero point arrays don't raise memoryview.cast(*) errors"""
        points_arr = self.makeArray(0)
        cloud_msg = rnp.msgify(PointCloud2, points_arr)
        new_points_arr = rnp.numpify(cloud_msg)

        np.testing.assert_equal(points_arr, new_points_arr)

if __name__ == '__main__':
    unittest.main()
