import numpy as np
from skimage import transform
import opensimplex
try:
    import napari
except ImportError:
    print('Could not import napari. Skipping visualization.')
    pass

def spherical2cartesian(R, T, P):
    x = R * np.cos(T) * np.sin(P)
    y = R * np.sin(T) * np.sin(P)
    z = R * np.cos(P)
    return x, y, z

def cartesian2spherical(x, y, z):
    R = np.sqrt(x**2 + y**2 + z**2)
    T = np.arctan2(y, x)
    P = np.arccos(z/R)
    return R, T, P

class NoiseBlob3D:
    default_size = 10
    def __init__(self, size):
        self.size = size

        self.mask = None
        self.texture = None

    def make_mask(self, gamma, circularity):
        contour, T, P = self.generate_contour(gamma, circularity)
        self.mask = self.generate_blob(contour, T, P).astype(bool)
        return self.mask

    def make_texture(self, gamma):
        if type(self.mask) == None: raise RuntimeError('Mask is not generated')
        self.texture = np.zeros(self.mask.shape)
        noise = self.generate_noise(self.mask.shape, gamma)
        noise -= noise.min()
        noise /= noise.max()
        self.texture[self.mask] = noise[self.mask]
        return self.texture

    def fill_contours(self, arr):
        return np.maximum.accumulate(arr, 1) * \
            np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] * \
            np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] * \
            np.maximum.accumulate(arr, 0)

    def generate_blob(self, contour, T, P):
        # Generate a mask
        i, j, k = spherical2cartesian(contour, T, P)
        i *= self.size
        j *= self.size
        k *= self.size

        i -= i.min()
        j -= j.min()
        k -= k.min()

        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)

        h = (i.max() + 1).astype(int)
        w = (j.max() + 1).astype(int)
        d = (k.max() + 1).astype(int)

        O = 10
        o = O//2

        mask = np.zeros([h+O, w+O, d+O])
        mask[i+o, j+o, k+o] = 1
        mask = self.fill_contours(mask)
        self.mask = mask

        return mask


    def generate_contour(self, gamma, circularity):
        # Creating a noise domain
        N = 100
        sampling_mult = 2
        A = self.generate_noise((N, N, N), gamma)

        # Sampling from noise domain
        c = N//2
        r = N//2-1
        t = np.linspace(0,np.pi,N*sampling_mult) # NOTE: Increase number of points, if there is a stringing artifact
        p = np.linspace(0,2*np.pi,N*sampling_mult) # NOTE: Increase number of points, if there is a stringing artifact
        r = np.ones_like(t) * r
        R, T, P = np.meshgrid(r, t, p)

        x, y, z = spherical2cartesian(R, T, P)
        x = (x + c).astype(int)
        y = (y + c).astype(int)
        z = (z + c).astype(int)
        noise_patch = A[x, y, z]

        # Adding noise to circle contour with weight circularity
        fundamental_mode = np.ones_like(noise_patch) * circularity
        contour = fundamental_mode + noise_patch
        contour /= contour.max()
        return contour, T, P

    def generate_noise(self, shape, gamma):
        N = max(shape)
        A = np.zeros(shape)
        # np.random.seed(5)
        for i in range(50):
            p = np.random.rand()*N*1.5
            y = np.linspace(p, p + i, shape[0])
            x = np.linspace(p, p + i, shape[1])
            z = np.linspace(p, p + i, shape[2])
            a = opensimplex.noise3array(z, x, y)*np.exp(-i*gamma)
            A += a

        A /= np.abs(A).max()
        return A

class Cell3D:
    """Class for generating a cell

    Raises:
        RuntimeError: When there is no way to position a nucleus within the
        cytoplasm

    Returns:
        Cell: self
    """
    blob_resolution = 30
    def __init__(self,  size_cytoplasm,
                        size_nucleus,
                        dx, dy, dz,
                        gamma_nucleus,
                        circ_nucleus,
                        gamma_cytoplasm,
                        circ_cytoplasm,
                        gamma_cytoplasm_texture,
                        lowest_intensity
                        ):
        """Cell generator, based on opensimplex noise generation

        Args:
            cytoplasm_size (float): diameter of cytoplasm in microns
            nucleus_size (float): diameter of nucleus in microns
            dx (float): pixel size in x direction in microns
            dy (float): pixel size in y direction in microns
            dz (float): pixel size in z direction in microns
            gamma_nucleus (float): noise parameter, defines the shape of nucleus. Typical values are between 0.5 and 1.5
            circ_nucleus (float): circularity of cytoplasm. Defines the shape of cytoplasm. Typical values are between 1 and 2
            gamma_cytoplasm (float): noise parameter, defines the shape of cytoplasm. Typical values are between 0.1 and 1.5
            circ_cytoplasm (float): circularity of cytoplasm. Defines the shape of cytoplasm. Typical values are between 1 and 3
            gamma_cytoplasm_texture (float): noise parameter, defines the frequency of cytoplasm texture. Typical values are between 0.01 and 2
            lowest_intensity (float): lowest intensity of cytoplasm. Typical values are between 0.1 and 0.5
            polygon_N (int): number of sides of polygonal blob
        """

        self.size_cytoplasm = size_cytoplasm 
        self.size_nucleus = size_nucleus 
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.gamma_nucleus = gamma_nucleus
        self.circ_nucleus = circ_nucleus
        self.gamma_cytoplasm = gamma_cytoplasm
        self.circ_cytoplasm = circ_cytoplasm
        self.gamma_cytoplasm_texture = gamma_cytoplasm_texture
        self.lowest_intensity = lowest_intensity

        self.mask_cytoplasm = None
        self.mask_nucleus = None
        self.texture_cytoplasm = None
        self.image = None

    def run(self):
        self.generate_cytoplasm()
        self.generate_nucleus()
        self.position_nucleus()
        self.apply_texture()
        self.scale_to_size()

    def _resize(self, arr:np.ndarray, size:float):
        h, w, d = arr.shape
        ar_xy = w / h
        ar_yx = h / w
        ar_zx = d / w
        x_out = size * ar_xy / self.dx
        y_out = size * ar_yx / self.dy
        z_out = size * ar_zx / self.dz
        return transform.resize(arr, (int(y_out), int(x_out), int(z_out)), anti_aliasing=False)
    
    def scale_to_size(self):
        self.mask_cytoplasm = self._resize(self.mask_cytoplasm, self.size_cytoplasm).astype(bool)
        self.image = self._resize(self.image, self.size_cytoplasm)
        self.cell_shape = self.image.shape

    def _normalize(self, arr):
        arr -= arr.min()
        arr /= arr.max()
        return arr

    def generate_nucleus(self):
        NB = NoiseBlob3D(size=self.blob_resolution * self.size_nucleus / self.size_cytoplasm)
        self.mask_nucleus = NB.make_mask(self.gamma_nucleus, self.circ_nucleus)

    def generate_cytoplasm(self):
        NB = NoiseBlob3D(size=self.blob_resolution)
        self.mask_cytoplasm = NB.make_mask(self.gamma_cytoplasm, self.circ_cytoplasm)
        self.texture_cytoplasm = NB.make_texture(self.gamma_cytoplasm_texture)

    def position_nucleus(self):
        terminator = 0
        while terminator <= 100:
            hc, wc, dc = self.mask_cytoplasm.shape
            hn, wn, dn = self.mask_nucleus.shape

            x = np.random.randint(0, wc-wn)
            y = np.random.randint(0, hc-hn)
            z = np.random.randint(0, dc-dn)

            ys = hc - hn - y
            xs = wc - wn - x
            zs = dc - dn - z

            shifted = np.pad(self.mask_nucleus, ((y,ys), (x,xs), (z,zs)), 'constant', constant_values=0)
            if np.all(self.mask_cytoplasm[shifted]):
                break
            else:
                terminator += 1
                if terminator == 100:
                    raise RuntimeError('Could not position nucleus, try to decrease nucleus size or increase cytoplasm size')

        self.mask_cytoplasm[shifted] = 0

    def show(self):
        viewer = napari.Viewer()
        viewer.add_image(self.image, colormap='gray', rendering='average')
        viewer.dims.ndisplay = 3
        napari.run()

    def apply_texture(self):
        self.image = np.zeros_like(self.texture_cytoplasm)
        self.image[self.mask_cytoplasm] = self.texture_cytoplasm[self.mask_cytoplasm] + self.lowest_intensity
        self.image = self._normalize(self.image)

if __name__ == '__main__':
    import napari

    dx = dy = 1
    dz = 1
    cell = Cell3D( 
        size_cytoplasm          = 30,
        size_nucleus            = 10,
        dx = dx, dy = dy, dz = dz,
        gamma_nucleus           = 1.5,
        circ_nucleus            = 2,
        gamma_cytoplasm         = 1,
        circ_cytoplasm          = 2.0,
        gamma_cytoplasm_texture = .1,
        lowest_intensity        = 0.1)

    print('make cell')
    cell.run()
    # np.save('cell.npy', cell.image)
    print('open napari')
    viewer = napari.Viewer()
    viewer.add_image(cell.image,
                        name='cell',
                        blending='additive',
                        colormap='gray',
                        rendering='mip')
    napari.run()