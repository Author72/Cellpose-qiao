MMD test-time adaptation
========================

Cellpose can perform conservative, foreground-aware Maximum Mean Discrepancy
test-time adaptation (MMD-TTA) for 2D images. It is intended for moderate domain
shifts such as contrast, noise, staining, and mild morphology changes. It does
not use target labels.

The implementation follows an episodic workflow:

1. Source-domain features are sampled from ground-truth cell pixels at the
   CPSAM ``encoder.neck`` output and saved as a compact bank.
2. The unadapted model produces a fixed high-confidence foreground map for each
   target image.
3. Multi-kernel MMD aligns target foreground features to the source bank. A
   photometric consistency loss and source-parameter anchor limit drift.
4. Only ``encoder.neck`` is updated. The flow/cell-probability readout and the
   rest of SAM remain frozen.
5. The adapted model predicts the current image and is then restored, so errors
   cannot accumulate across unrelated images.

Building a source bank
----------------------

Use the same trained model and preprocessing settings that will be used at
inference. ``source_masks`` are 2D instance-label arrays; every nonzero pixel is
treated as cell foreground.

.. code-block:: python

   from cellpose import io, models

   model = models.CellposeModel(
       gpu=True,
       pretrained_model="/path/to/your_model",
   )
   source_images = [io.imread(path) for path in source_image_paths]
   source_masks = [io.imread(path) for path in source_mask_paths]

   bank = model.build_mmd_source_bank(
       source_images,
       source_masks,
       max_features=10000,
       batch_size=8,
   )
   bank.save("source_mmd_bank.pt")

The bank contains detached feature vectors and metadata, not source images or
model weights.

Python inference
----------------

.. code-block:: python

   from cellpose import io, models
   from cellpose.mmd_tta import MMDTTAConfig

   model = models.CellposeModel(
       gpu=True,
       pretrained_model="/path/to/your_model",
   )
   model.load_mmd_source_bank("source_mmd_bank.pt")

   config = MMDTTAConfig(
       steps=3,
       learning_rate=1e-5,
       foreground_threshold=0.8,
       bandwidths=(0.5, 1.0, 2.0, 4.0),
       mmd_weight=0.1,
       consistency_weight=0.1,
       anchor_weight=1e-4,
   )
   masks, flows, styles = model.eval(
       io.imread("target.tif"),
       mmd_tta=config,
   )

``model.last_mmd_tta_history`` records the component losses and sample count for
each successful update. If a tile contains fewer than ``min_samples`` confident
foreground features, it is skipped; if every tile is skipped, ordinary source
model inference is used.

Command-line inference
----------------------

After creating the bank through Python, enable episodic adaptation with:

.. code-block:: console

   cellpose --dir /path/to/target_images \
     --pretrained_model /path/to/your_model \
     --mmd_source_bank source_mmd_bank.pt \
     --mmd_tta_steps 3 --mmd_tta_lr 1e-5 \
     --mmd_tta_threshold 0.8 --save_tif

Practical guidance
------------------

Start with one to three steps and compare against the unadapted model on a held
out labelled target subset. MMD becoming smaller does not by itself prove that
segmentation improved. For a genuine new morphology absent from the source
domain, keep ``mmd_weight`` small; strong alignment can erase useful target-only
structure. MMD-TTA currently rejects ``do_3D=True`` and should be evaluated
separately before use on continual video streams.
