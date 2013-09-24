import nibabel
import vtk
import numpy as np
from scipy import ndimage

def vtk_iso(vol, iso_thresh=1):
    im_data = vol.tostring()
    img = vtk.vtkImageImport()
    img.CopyImportVoidPointer(im_data, len(im_data))
    img.SetDataScalarType(vtk.VTK_UNSIGNED_SHORT)
    img.SetNumberOfScalarComponents(1)
    img.SetDataExtent(0, vol.shape[2]-1, 0, vol.shape[1]-1, 0, vol.shape[0]-1)
    img.SetWholeExtent(0, vol.shape[2]-1, 0, vol.shape[1]-1, 0, vol.shape[0]-1)
    iso = vtk.vtkMarchingCubes()
    iso.SetInput(img.GetOutput())
    iso.SetValue(0, iso_thresh)
    return iso,img

def vtk_smooth(iso, iter=20, relax=0.5, decimate=0.0):
    isoSmooth = vtk.vtkSmoothPolyDataFilter()
    if decimate>0:
        deci = vtk.vtkDecimatePro()
        deci.SetInput(iso.GetOutput())
        deci.SetTargetReduction(decimate)
        deci.PreserveTopologyOn()
        isoSmooth.SetInputConnection(deci.GetOutputPort())
    else:
        isoSmooth.SetInputConnection(iso.GetOutputPort())
    isoSmooth.SetNumberOfIterations(100)
    isoSmooth.BoundarySmoothingOn()
    isoSmooth.FeatureEdgeSmoothingOff()
    isoSmooth.SetFeatureAngle(45)
    isoSmooth.SetEdgeAngle(15)
    isoSmooth.SetRelaxationFactor(relax)
    return isoSmooth

def vtk_render_window(iso, img=None, color=[0.5,0.5,0.5]):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(iso.GetOutputPort())
    normals.FlipNormalsOn()
    isoMapper = vtk.vtkPolyDataMapper()
    isoMapper.SetInput(normals.GetOutput())
    isoMapper.ScalarVisibilityOff()

    isoActor = vtk.vtkActor()
    isoActor.SetMapper(isoMapper)
    isoActor.GetProperty().SetColor(color)

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors to the renderer, set the background and size
    if img!=None:
        outline = vtk.vtkOutlineFilter()
        outline.SetInput(img.GetOutput())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInput(outline.GetOutput())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)
        outlineActor.VisibilityOff()
        ren.AddActor(outlineActor)

    ren.AddActor(isoActor)
    ren.SetBackground(1.0,1.0,1.0)
    renWin.SetSize(450,450)
    ## ren.GetActiveCamera().Elevation(235)
    ## ren.GetActiveCamera().SetViewUp(0,.5,-1)
    ## ren.GetActiveCamera().Azimuth(90)
    iren.Initialize()
    return iren

def montage(vol, ncols=None):
    """Returns a 2d image monage given a 3d volume."""
    ncols = ncols if ncols else int(np.ceil(np.sqrt(vol.shape[2])))
    rows = np.array_split(vol, range(ncols,vol.shape[2],ncols), axis=2)
    # ensure the last row is the same size as the others
    rows[-1] = np.dstack((rows[-1], np.zeros(rows[-1].shape[0:2] + (rows[0].shape[2]-rows[-1].shape[2],))))
    im = np.vstack([np.squeeze(np.hstack(np.dsplit(row, ncols))) for row in rows])
    return(im)

# get the pial surface
brainmask = nibabel.load('t1_brain_mask.nii.gz')
brainmask_vol = brainmask.get_data()
pial_iso,pial_img = vtk_iso(brainmask_vol)

# get the white matter surface
t1 = nibabel.load('t1.nii.gz')
t1_vol = t1.get_data()
# apply the brain mask to remove the scalp
t1_vol[brainmask_vol==0] = 0
# segment the white matter
wm_vol = (t1_vol>7000).astype(np.float)
wm_vol = ndimage.gaussian_filter(wm_vol.astype(float), 0.3)
wm_vol = ndimage.measurements.label(wm_vol)[0]
wm_vol = (wm_vol==1).astype(np.uint16)

imshow(montage(t1_vol[:,:,20:2:200]), cmap='gray')
show()

wm_iso,wm_img = vtk_iso(wm_vol)
wm_iso = vtk_smooth(wm_iso, 20, 0.3, 0.2)

iren = vtk_render_window(wm_iso)
iren.Start()

