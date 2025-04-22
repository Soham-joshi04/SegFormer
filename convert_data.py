import os
import numpy as np
import nibabel as nib

# CONFIGURATION — adjust these paths & modalities as needed
BRATS_ROOT = '/path/to/MICCAI_BraTS2020_TrainingData'
OUT_ROOT   = '/path/to/segformer_data_npy'
MODALITIES = ['flair', 't1', 't1ce', 't2']   # you can pick all or a subset
SLICE_AXIS = 2                              # axial: index 2 in (240,240,155)

# make output dirs
images_out = os.path.join(OUT_ROOT, 'imagesTr')
masks_out  = os.path.join(OUT_ROOT, 'labelsTr')
os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out,  exist_ok=True)

for case in sorted(os.listdir(BRATS_ROOT)):
    case_dir = os.path.join(BRATS_ROOT, case)
    # load all modalities into a dict of 3D arrays
    vols = {m: nib.load(os.path.join(case_dir, f"{case}_{m}.nii.gz")).get_fdata()
            for m in MODALITIES}
    # load segmentation map
    seg  = nib.load(os.path.join(case_dir, f"{case}_seg.nii.gz")).get_fdata().astype(np.uint8)

    # number of slices along the SLICE_AXIS
    D = vols[MODALITIES[0]].shape[SLICE_AXIS]

    for z in range(D):
        # extract the 2D mask for this slice
        mask_slice = seg.take(indices=z, axis=SLICE_AXIS)
        # Optionally skip blank slices:
        if mask_slice.max() == 0:
            continue

        # stack the same slice index from each modality → shape (M, H, W)
        img_stack = np.stack([
            vols[m].take(indices=z, axis=SLICE_AXIS)
            for m in MODALITIES
        ], axis=0).astype(np.float32)

        # save .npy files
        img_fname  = f"{case}_slice{z:03d}.npy"
        mask_fname = f"{case}_slice{z:03d}.npy"
        np.save(os.path.join(images_out, img_fname),  img_stack)
        np.save(os.path.join(masks_out,  mask_fname), mask_slice)

print("Done! Data layout under:", OUT_ROOT)
print("  imagesTr/:   (num_modalities, H, W) .npy per slice")
print("  labelsTr/:    (H, W) .npy per slice")
