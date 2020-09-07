import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.optimize as opt
from scipy import ndimage

def myshow(img, title='', transpose=False):
    nda = sitk.GetArrayViewFromImage(img)
    if transpose:
        plt.imshow(nda.T)
    else:
        plt.imshow(nda)
    plt.title(title)


def downsample(image, factor=2):
    """Downsample the input sitk image by a factor (greater than 1).
    Parameters
    image: sitk Image
        input image
    factor: float
        downsampling factor between greater than 1 (default 2)
    """
    assert factor > 1, "Error: downsampling factor value must be larger than 1."
    old_size = np.array(image.GetSize())
    new_size = (1.0 * old_size / factor).astype(int).tolist()

    old_spacing = np.array(image.GetSpacing())
    new_spacing = (old_spacing * factor).tolist()
    default_value = 0
    resampleSliceFilter = sitk.ResampleImageFilter()
    new_image = resampleSliceFilter.Execute(image, new_size, sitk.Transform(),
                                                sitk.sitkLinear, image.GetOrigin(),
                                                new_spacing, image.GetDirection(),
                                                default_value, image.GetPixelID())
    return new_image


def resample_isotropic(image, default_val=0, interpolator=sitk.sitkLinear):
    """Resample simpleITK image to isotropic pixel size
    Parameters
        image: simpleITK Image
        default_val: float
            Default value for resampling filter (for empty regions)
        interpolator: simpleITK interpolator
            Can be one of:
                sitk.sitkLinear (default),
                sitk.sitkBSpline,
                sitk.sitkGaussian,
                sitk.sitkHammingWindowedSinc
                sitk.sitkBlackmanWindowedSinc,
                sitk.sitkCosineWindowedSinc,
                sitk.sitkWelchWindowedSinc,
                sitk.sitkLanczosWindowedSinc.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    min_spacing = min(image.GetSpacing())
    new_spacing = [min_spacing] * image.GetDimension()
    new_size = np.asarray(original_size) * np.asarray(original_spacing) / min_spacing
    new_size = np.round(new_size).astype(int).tolist()
    resampleSliceFilter = sitk.ResampleImageFilter()
    img_resampled = resampleSliceFilter.Execute(image, new_size, sitk.Transform(),
                                                interpolator, image.GetOrigin(),
                                                new_spacing, image.GetDirection(),
                                                default_val, image.GetPixelID())
    return img_resampled


def resample_moving2fixed(fixed_image, moving_image, default_val=0):
    """Resample the moving image to the same spacing as fixed image.
    Moving image outside of fixed image boundaries is cropped.
    Origin of the fixed_image is used for output."""
    assert fixed_image.GetSpacing() == moving_image.GetSpacing(), "Error: spacing must be equal between moving and fixed image"

    transform = sitk.Transform(fixed_image.GetDimension(), sitk.sitkIdentity)

    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetDefaultPixelValue(default_val)
    filter_resample.SetTransform(transform)
    filter_resample.SetOutputSpacing(fixed_image.GetSpacing())
    filter_resample.SetInterpolator(sitk.sitkLinear)
    filter_resample.SetOutputOrigin(fixed_image.GetOrigin())
    filter_resample.SetOutputDirection(fixed_image.GetDirection())
    filter_resample.SetSize(fixed_image.GetSize())
    resampled_image = filter_resample.Execute(moving_image)
    return resampled_image


def threshold_based_crop(image, padding_px=80):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.

    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.

    Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/70_Data_Augmentation.html
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    bb_starting_ind = max(bounding_box[0] - padding_px, 0), \
                max(bounding_box[1] - padding_px, 0), \
                max(bounding_box[2] - padding_px, 0)

    bb_size = min(bounding_box[3] + 2 * padding_px, image.GetWidth() - bb_starting_ind[0]), \
            min(bounding_box[4] + 2 * padding_px, image.GetHeight() - bb_starting_ind[1]), \
            min(bounding_box[5] + 2 * padding_px, image.GetDepth() - bb_starting_ind[2])
    cropped_image = sitk.RegionOfInterest(image, bb_size, bb_starting_ind)
    return cropped_image, bb_size, bb_starting_ind


def normalize_8bit(image_array, top_perc=99.995):
    """Normalize image intensity so that top 99.99% percentile is mapped to 255 and bottom 1% to 0.
    This is computationally expensive for large images.
    Parameters:
    image_array: array_like
        Input array.
    top_perc: float
        The top percentile of cut-off (default 99.995)

    Returns:
    im_rescaled: ndarray
        Rescaled array of type 'uint8'.
    """
    im_low = np.percentile(image_array, 1)
    im_high = np.percentile(image_array, top_perc)
    im_clipped = np.clip(image_array, im_low, im_high)
    im_rescaled = (255.0 * (im_clipped - im_low) / (1.0*im_high - im_low)).astype('uint8')
    return im_rescaled


def to_8bit(image):
    """Rescale image intensity to [0,255] and convert image to 8-bit
    Parameters
        image: sitk Image

    Return
        sitk Image of type sitkUInt8.
    """
    new_img = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)
    return new_img

def normalize(image, method='max'):
    """Normalize image by maximum or sum.
    Parameters
        image: sitk Image
        method: str 
            'max', make max value = 1
            'sum', make sum of intensities =1
    Returns
        sitk.Image()
        Normalized image of type 'sitkFloat32'.
        """
    img_float32 = sitk.Cast(image, sitk.sitkFloat32)
    if method == 'max':
        minv, maxv = get_minmax(image)
        image_normalized = img_float32 / maxv
    elif method == 'sum':
        image_normalized = img_float32 / get_sum(image)
    else:
        image_normalized = None
        print("Normalization method unknown\n")
    return image_normalized


def get_sum(image):
    im_array = sitk.GetArrayFromImage(image)
    return im_array.sum()

def get_minmax(image):
    filter_minmax = sitk.MinimumMaximumImageFilter()
    filter_minmax.Execute(image)
    minv, maxv = filter_minmax.GetMinimum(), filter_minmax.GetMaximum()
    return minv, maxv


def show_mips(image, title='', boundary_width=2, cmap='gray', scalebar=None, crosshair=False):
    """
    Parameters:
    scalebar: (float), size of scalebar in microns, default None.
    """
    if image.GetDimension() == 3:
        mip_x_original_spacing = sitk.MaximumProjection(image, 0)[0, :, :]
        mip_y_original_spacing = sitk.MaximumProjection(image, 1)[:, 0, :]
        mip_z_original_spacing = sitk.MaximumProjection(image, 2)[:, :, 0]

        mip_x = resample_isotropic(mip_x_original_spacing)
        mip_y = resample_isotropic(mip_y_original_spacing)
        mip_z = sitk.PermuteAxes(resample_isotropic(mip_z_original_spacing), [1, 0])

        tiled = sitk.Tile(mip_x, mip_y, mip_z, [2, 2])
        tiled_array = sitk.GetArrayFromImage(tiled)
        # add bright boundary
        boundary_value = tiled_array.max()
        tiled_array[mip_x.GetSize()[1] - boundary_width: mip_x.GetSize()[1], :] = boundary_value
        tiled_array[:, mip_x.GetSize()[0] - boundary_width: mip_x.GetSize()[0]] = boundary_value
        # show monochrome image in selected color map
        if image.GetNumberOfComponentsPerPixel() == 1:
            plt.imshow(tiled_array, aspect='equal', cmap=cmap)
        else: # RGB colormap for 3-component images
            plt.imshow(tiled_array, aspect='equal')
        # add axis labels
        plt.text(0.05 * mip_x.GetSize()[0], 0.95 * mip_x.GetSize()[1], 'YZ', color='w', fontSize=14)
        plt.text(0.05 * mip_x.GetSize()[0], mip_x.GetSize()[1] + 0.95 * mip_z.GetSize()[1], 'XY', color='w', fontSize=14)
        plt.text(mip_x.GetSize()[0] + 0.05 * mip_y.GetSize()[0], 0.95 * mip_x.GetSize()[1], 'XZ', color='w', fontSize=14)
        # add scalebar
        if scalebar is not None:
            scalebar_px = scalebar/mip_x.GetSpacing()[0]
            plt.plot([0.95 * mip_x.GetSize()[0] - scalebar_px, 0.95 * mip_x.GetSize()[0]], 
                     [0.95 * mip_x.GetSize()[1], 0.95 * mip_x.GetSize()[1]], color='w', lw='2')
        # add crosshair: toDo
    else: # 2D image
        assert image.GetSpacing()[0] == image.GetSpacing()[1], "Image spacing must be the same in X and Y."
        tiled_array = sitk.GetArrayFromImage(image)
        # show monochrome image in selected color map
        if image.GetNumberOfComponentsPerPixel() == 1:
            plt.imshow(tiled_array, aspect='equal', cmap=cmap)
        else: # RGB colormap for 3-component images
            plt.imshow(tiled_array, aspect='equal')
        if scalebar is not None:
            scalebar_px = scalebar/image.GetSpacing()[0]
            plt.plot([0.95 * image.GetSize()[0] - scalebar_px, 0.95 * image.GetSize()[0]], 
                     [0.95 * image.GetSize()[1], 0.95 * image.GetSize()[1]], color='w', lw='2')
        if crosshair:
            plt.plot([-0.5 + 0.5*image.GetSize()[0], -0.5 + 0.5*image.GetSize()[0]], 
                     [0.05 * image.GetSize()[1], 0.95 * image.GetSize()[1]], color='y', lw='1')
            plt.plot([0.05*image.GetSize()[0], 0.95*image.GetSize()[0]], 
                     [-0.5 + 0.5 * image.GetSize()[1], -0.5 + 0.5 * image.GetSize()[1]], color='y', lw='1')
    plt.title(title)


def transform_resample(image, transform, default_val=0, origin='new'):
    """Transform the image and resample to the original grid."""
    extreme_points = [image.TransformIndexToPhysicalPoint((0, 0, 0)),
                      image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, 0)),
                      image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), 0)),
                      image.TransformIndexToPhysicalPoint((0, image.GetHeight(), 0)),
                      image.TransformIndexToPhysicalPoint((0, 0, image.GetDepth())),
                      image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, image.GetDepth())),
                      image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), image.GetDepth())),
                      image.TransformIndexToPhysicalPoint((0, image.GetHeight(), image.GetDepth()))]
    inv_transform = transform.GetInverse()

    extreme_points_transformed = [inv_transform.TransformPoint(pnt) for pnt in extreme_points]
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]

    max_x = max(extreme_points_transformed)[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    # Use the original spacing (arbitrary decision).
    output_spacing = image.GetSpacing()
    # Identity cosine matrix (arbitrary decision).
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # Minimal x,y,z coordinates are the new origin.
    if origin == 'new':
        output_origin = [min_x, min_y, min_z]
    else:
        output_origin = image.GetOrigin()
    # Compute grid size based on the physical size and spacing.
    output_size = [round((max_x - min_x) / output_spacing[0]),
                   round((max_y - min_y) / output_spacing[1]),
                   round((max_z - min_z) / output_spacing[2])]
    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetDefaultPixelValue(default_val)
    filter_resample.SetTransform(transform)
    filter_resample.SetOutputSpacing(output_spacing)
    filter_resample.SetInterpolator(sitk.sitkLinear)
    filter_resample.SetOutputOrigin(output_origin)
    filter_resample.SetOutputDirection(output_direction)
    filter_resample.SetSize(output_size)
    resampled_image = filter_resample.Execute(image)
    return resampled_image


def merge_images_rgb(image_fixed, image_moving):
    """Create RGB image from two input images for colocalization visualization.
    The resulting image contains first image in green, second image in magenta, and their overlap in white.
    Parameters:
    image_fixed : sitk Image
        Input image (mono), converted into green channel.
    image_moving : sitk Image
        Input image (mono), converted into magenta channel.

    Returns:
    img_merge_rgb: sitk Image, RGB.
    """
    assert image_fixed.GetSize() == image_moving.GetSize(), "Image sizes must be equal. Did you forget resampling step?"
    image_fixed_rescaled = sitk.Cast(sitk.RescaleIntensity(image_fixed, 0, 255), sitk.sitkUInt8)
    image_moving_rescaled = sitk.Cast(sitk.RescaleIntensity(image_moving, 0, 255), sitk.sitkUInt8)
    arr_fixed = sitk.GetArrayFromImage(image_fixed_rescaled)
    arr_moving = sitk.GetArrayFromImage(image_moving_rescaled)
    dim = image_fixed.GetDimension()
    if dim == 3:
        arr_merge_rgb = np.zeros((arr_fixed.shape[0], arr_fixed.shape[1], arr_fixed.shape[2], 3), dtype='uint8')
        # fill Green ch (fixed image)
        arr_merge_rgb[:, :, :, 1] = arr_fixed
        # fill Magenta ch (moving image)
        arr_merge_rgb[:, :, :, 0] = arr_merge_rgb[:, :, :, 2] = arr_moving
    else: #2d case
        arr_merge_rgb = np.zeros((arr_fixed.shape[0], arr_fixed.shape[1], 3), dtype='uint8')
        # fill Green ch (fixed image)
        arr_merge_rgb[:, :, 1] = arr_fixed
        # fill Magenta ch (moving image)
        arr_merge_rgb[:, :, 0] = arr_merge_rgb[:, :, 2] = arr_moving

    img_merge_rgb = sitk.GetImageFromArray(arr_merge_rgb, isVector=True)
    img_merge_rgb.SetSpacing(image_fixed.GetSpacing())
    return img_merge_rgb

      
def get_center(image, method='geometry', bg_percentile=99):
    """Set the sitk Image origin to it's geometric or center of mass center
    Parameters
        image: sitk Image
            Image is changed in-place by reference.
        method: str
            'geometry' (default), use the geometrical center of input image as new origin
            'mass', use center-of-mass method
        bg_percentile: float
            if method='mass', use this number to determine image background threshold (0 to 100)
    Returns
        None, the input image is changed by reference.
    """
    spacing_xyz = image.GetSpacing()
    if method == 'geometry':
        new_origin = np.asarray(image.GetSize()) / 2.0 * np.array(spacing_xyz)
        return new_origin
    elif method == 'mass':
        im_array = sitk.GetArrayFromImage(image)
        bg = np.percentile(im_array, bg_percentile)
        mask = im_array > bg
        cmass_xyz = list(ndimage.measurements.center_of_mass(mask))
        cmass_xyz.reverse()
        new_origin = np.array(spacing_xyz) * (np.array(cmass_xyz) + 0.5)
        return new_origin
    else:
        raise ValueError('Recentering method is unknown.')


def save_transform(transform, filename):
    """Save transform to tfm file"""
    sitk.WriteTransform(transform, filename + '.tfm')


def gaussian_3d(x_y_z, xo, yo, zo, sigma_x, sigma_y, sigma_z, amplitude, offset, ravel=True):
    """"Generate 3D Gaussian function. Takes arguments in xyz order, returns function in zyx order."""
    x, y, z = x_y_z
    g = offset + amplitude * np.exp(- (((x - xo) ** 2) / (2.0 * sigma_x ** 2)
                                       + ((y - yo) ** 2) / (2.0 * sigma_y ** 2)
                                       + ((z - zo) ** 2) / (2.0 * sigma_z ** 2)))
    g = np.transpose(g)
    if ravel:
        g = g.ravel()

    return g

def gaussian_2d(x_y, xo, yo, sigma_x, sigma_y, amplitude, offset, ravel=True):
    """"Generate 2D Gaussian function. Takes arguments in xy order, returns function in yx order."""
    x, y = x_y
    g = offset + amplitude * np.exp(- (((x - xo) ** 2) / (2.0 * sigma_x ** 2)
                                       + ((y - yo) ** 2) / (2.0 * sigma_y ** 2)))
    g = np.transpose(g)
    if ravel:
        g = g.ravel()

    return g

def get_FWHM(img, output_units='um', debug_mode=False, dim=3):
    """Get FWHM(x,y,z) of PSF by 3D Gaussian fitting.
    Parameters
        image: sitk Image
            Input image containing a single bright spot (PSF) near the center.
        output_units: str
            Output units, 'um' (default) or 'px'.
        debug_mode: bool
            If True, print out all fitted values (default False).
        dim: int
            image dimension, default 3.
    Returns
        FWHM_xyz: tuple
        The FWHM(x,y,z) in um or px, depending on selected output units.
    """
    voxel_size = list(img.GetSpacing())
    assert all(x == voxel_size[0] for x in voxel_size), "Error: image spacing (voxel dimension) must be isotropic"
    if dim == 2:
        assert img.GetDimension() == 2, "Error: image dimension must be 2."
    else:
        assert img.GetDimension() == 3, "Error: image dimension must be 3."

    # normalize image intensity:
    img_norm = sitk.RescaleIntensity(sitk.Cast(img, sitk.sitkFloat32), 0.0, 1.0)
    # switch to numpy array and pixel units
    im_array = sitk.GetArrayFromImage(img_norm)

    if dim == 3:
        x = np.linspace(0, im_array.shape[2] - 1, im_array.shape[2]) + 0.5
        y = np.linspace(0, im_array.shape[1] - 1, im_array.shape[1]) + 0.5
        z = np.linspace(0, im_array.shape[0] - 1, im_array.shape[0]) + 0.5
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    else:
        x = np.linspace(0, im_array.shape[1] - 1, im_array.shape[1]) + 0.5
        y = np.linspace(0, im_array.shape[0] - 1, im_array.shape[0]) + 0.5
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    # estimate the center position
    peak_val = im_array.max()
    max_ind_array = np.where(im_array == peak_val)
    if dim == 3:
        (z_peak, y_peak, x_peak) = max_ind_array[0][0], max_ind_array[1][0], max_ind_array[2][0]
    else:
        (y_peak, x_peak) = max_ind_array[0][0], max_ind_array[1][0]

    # Parameters: xpos, ypos, zpos, sigmaX, sigmaY, sigmaZ, amplitude, offset
    if dim == 3:
        initial_guess = (x_peak, y_peak, z_peak, 1.0, 1.0, 1.0, 1.0, 0.0)
        popt, pcov = opt.curve_fit(gaussian_3d, (x_grid, y_grid, z_grid),
                                   im_array.ravel(), p0=initial_guess,
                                   bounds=(
                                   (im_array.shape[2] * 0.25, im_array.shape[1] * 0.25, im_array.shape[0] * 0.25,
                                    0.0, 0.0, 0.0,
                                    0.8, -0.1),
                                   (im_array.shape[2] * 0.75, im_array.shape[1] * 0.75, im_array.shape[0] * 0.75,
                                    im_array.shape[2] / 6.0, im_array.shape[1] / 6.0, im_array.shape[0] / 6.0,
                                    1.2, 0.1)))

        xcenter, ycenter, zcenter, sigmaX, sigmaY, sigmaZ, amp, offset = popt
        if debug_mode:
            print('\n Fitted values: xcenter, ycenter, zcenter, sigmaX, sigmaY, sigmaZ, amp, offset:\n')
            print(np.array2string(popt, separator=',  ', floatmode='fixed'))
        FWHM_xyz = 2 * np.asarray([sigmaX, sigmaY, sigmaZ]) * np.sqrt(2 * np.log(2))
        if output_units == 'um':
            FWHM_xyz = FWHM_xyz * np.asarray(img.GetSpacing())
    else:
        initial_guess = (x_peak, y_peak, 1.0, 1.0, 1.0, 0.0)
        popt, pcov = opt.curve_fit(gaussian_2d, (x_grid, y_grid),
                                   im_array.ravel(), p0=initial_guess,
                                   bounds=(
                                   (im_array.shape[1] * 0.25, im_array.shape[0] * 0.25,
                                    0.0, 0.0, 
                                    0.8, -0.1),
                                   (im_array.shape[1] * 0.75, im_array.shape[0] * 0.75,
                                    im_array.shape[1] / 6.0, im_array.shape[0] / 6.0,
                                    1.2, 0.1)))

        xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt
        if debug_mode:
            print('\n Fitted values: xcenter, ycenter, sigmaX, sigmaY,  amp, offset:\n')
            print(np.array2string(popt, separator=',  ', floatmode='fixed'))
        FWHM_xyz = 2 * np.asarray([sigmaX, sigmaY]) * np.sqrt(2 * np.log(2))
        if output_units == 'um':
            FWHM_xyz = FWHM_xyz * np.asarray(img.GetSpacing())
            
    return FWHM_xyz