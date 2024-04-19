# XRD corrections
# This class aims to encapsulate the XRD corrections performed during experiment MECL10034
# Written by Thomas Stevens
# With assistance by Adrien Descamps and Celine Crepisson

import numpy as np


class FullCorrections:
    def __init__(self, poni, calib_tif):
        self.coord_sample = None
        self.id1 = None
        self.id2 = None
        self.params = None
        self.PONI = poni
        self.CALIB = calib_tif
        return

    def SOLID_ANGLE(self):
        out = self.PONI.solidAngleArray(absolute=True)
        return out

    def FILTER(self, thickness, atten_len):  # atten_length and thickness in same units
        d2, d1 = np.meshgrid(np.arange(0, self.CALIB.shape[1], 1), np.arange(0, self.CALIB.shape[0], 1))
        cosIncidence = self.PONI.cos_incidence(d1, d2)
        out = np.exp(-(thickness / cosIncidence) / atten_len)
        return out, cosIncidence

    def SELF_ATTENUATION(self, atten_in, atten_out, alpha, D):
        cos_alpha = np.cos(alpha * np.pi / 180)
        normalZ = cos_alpha
        normalY = -np.sin(alpha * np.pi / 180)
        normalX = 0
        normalSurface = np.array([[normalX, normalY, normalZ]])
        cos_beta = np.dot(normalSurface, self.coord_sample / np.linalg.norm(self.coord_sample, axis=0))
        cos_beta_reshape = np.reshape(cos_beta, self.CALIB.shape)
        chi = (1 / (atten_in * cos_alpha)) - (1 / (atten_out * cos_beta))
        out = np.exp(-D * (cos_beta >= 0) / (cos_beta * atten_out)) * (1 - np.exp(-chi * D)) / (cos_alpha * chi)
        # cos_beta >= 0 to account for transmission and reflection geometry
        out = np.reshape(out, self.CALIB.shape)
        return out

    def cos_beta(self, alpha):
        cos_alpha = np.cos(alpha * np.pi / 180)
        normalZ = cos_alpha
        normalY = -np.sin(alpha * np.pi / 180)
        normalX = 0
        normalSurface = np.array([[normalX, normalY, normalZ]])
        cos_beta = np.dot(normalSurface, self.coord_sample / np.linalg.norm(self.coord_sample, axis=0))
        cos_beta_reshape = np.reshape(cos_beta, self.CALIB.shape)
        return cos_beta_reshape

    def rotation_matrix(self, param=None):
        """Compute and return the detector tilts as a single rotation matrix
        Corresponds to rotations about axes 1 then 2 then 3 (=> 0 later on)
        For those using spd (PB = Peter Boesecke), tilts relate to
        this system (JK = Jerome Kieffer) as follows:
        JK1 = PB2 (Y)
        JK2 = PB1 (X)
        JK3 = PB3 (Z)
        ...slight differences will result from the order
        FIXME: make backwards and forwards converter helper function
        axis1 is  vertical and perpendicular to beam
        axis2 is  horizontal and perpendicular to beam
        axis3  is along the beam, becomes axis0
        see:
        https://pyfai.readthedocs.io/en/latest/geometry.html#detector-position
        or ../doc/source/img/PONI.png
        :param param: list of geometry parameters, defaults to self.param
                      uses elements [3],[4],[5]
        :type param: list of float
        :return: rotation matrix
        :rtype: 3x3 float array
        """
        if param is None:
            param = self.params
        cos_rot1 = np.cos(param[3])
        cos_rot2 = np.cos(param[4])
        cos_rot3 = np.cos(param[5])
        sin_rot1 = np.sin(param[3])
        sin_rot2 = np.sin(param[4])
        sin_rot3 = np.sin(param[5])

        # Rotation about axis 1: Note this rotation is left-handed
        rot1 = np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_rot1, sin_rot1],
                         [0.0, -sin_rot1, cos_rot1]])
        # Rotation about axis 2. Note this rotation is left-handed
        rot2 = np.array([[cos_rot2, 0.0, -sin_rot2],
                         [0.0, 1.0, 0.0],
                         [sin_rot2, 0.0, cos_rot2]])
        # Rotation about axis 3: Note this rotation is right-handed
        rot3 = np.array([[cos_rot3, -sin_rot3, 0.0],
                         [sin_rot3, cos_rot3, 0.0],
                         [0.0, 0.0, 1.0]])
        rotation_matrix = np.dot(np.dot(rot3, rot2),
                                 rot1)  # 3x3 matrix

        return rotation_matrix

    def pixel_positions(self):
        # wavelength = self.PONI.get_wavelength() * 1e10
        dist, poni1, poni2, rot1, rot2, rot3 = (self.PONI.get_dist(), self.PONI.get_poni1(), self.PONI.get_poni2(),
                                                self.PONI.get_rot1(), self.PONI.get_rot2(), self.PONI.get_rot3())
        self.params = np.array([dist, poni1, poni2, rot1, rot2, rot3])

        # Get pixel position in real units in detector frame

        pixelSize = 150e-6  # in m

        id2, id1 = np.arange(0, self.CALIB.shape[1], 1) * pixelSize, np.arange(0, self.CALIB.shape[0], 1) * pixelSize
        self.id2, self.id1 = id2, id1
        p2_pixel, p1_pixel = np.meshgrid(id2, id1)
        p3_pixel = None

        # Get pixel position in real units in lab frame

        size = p1_pixel.size
        p2 = (p2_pixel - poni2).ravel()  # axis 2: horizontal and perpendicular to beam
        p1 = (p1_pixel - poni1).ravel()  # axis 1: vertical and perpendicular to beam

        assert size == p2_pixel.size

        # note the change of sign in the third dimension:
        # Before the rotation we are in the detector's referential,
        # the sample position is at d3= -L <0
        # the sample detector distance is always positive.
        if p3_pixel is None:
            p3 = np.zeros(size) + dist
        else:
            p3 = (dist + p3_pixel).ravel()
            assert size == p3.size
        print(np.shape(p1), np.shape(p2), np.shape(p3))

        # Pixel position in the detector frame centered at the sample position
        coord_det = np.vstack((p1, p2, p3))  # To get the same orientation as pyFAI for the rotation (axis1, axis2,
        # axis3) triad (vertical axis, horizontal axis, along X-ray beam)

        # Pixel position in the sample frame
        self.coord_sample = np.dot(self.rotation_matrix(), coord_det)
        return
