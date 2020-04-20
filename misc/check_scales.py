import h5py
import napari


def check_scale(scale):
    path = '../antibodies/test-instance-analysis2/WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.h5'
    with h5py.File(path, 'r') as f:
        raw = f['serum/s%i' % scale][:]
        seg = f['cell_segmentation/s%i' % scale][:]
        nuclei = f['nucleus_segmentation/s%i' % scale][:]
    print(raw.shape)
    print(seg.shape)
    print(nuclei.shape)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_labels(nuclei, name='seeds')
        viewer.add_labels(seg, name='cells')


check_scale(1)
