Command line
------------------------

The full list of options and what they do can be found on the Command Line Interface (CLI) documentation
page: :ref:`Cellpose CLI`. A description of the most important settings can be found on the :ref:`Settings` page.

.. _Command line examples:

Command Line Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``python -m cellpose`` and specify parameters as below. For instance
to run on a folder with images where cytoplasm is green and nucleus is
blue and save the output as a png (using default diameter 30):

::

   python -m cellpose --dir /home/carsen/images_cyto/test/ --save_png

To run on a single 3D image:

:: 
   
   python -m cellpose --image_path /home/carsen/image3D.tif --do_3D --flow3D_smooth 2 --save_tif


.. warning:: 
    The path given to ``--dir`` is recommended to be an absolute path.
Flow test-time adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~

For 2D images whose cell appearance differs substantially from the training
data, Cellpose can adapt the final flow readout on each test image without
labels. The objective enforces that Y/X flow vectors transform correctly under
horizontal and vertical flips. The original checkpoint is never modified: the
adapted readout is restored immediately after processing each image.

Enable it conservatively with a small number of steps::

    python -m cellpose --dir /path/to/images --tta_steps 8 --tta_lr 1e-4

``--tta_batch_size`` controls the number of 256-pixel crops used for the
adaptation (default: 1; increase only when GPU memory permits), and ``--tta_confidence`` excludes lower-confidence
pixels from its self-supervised loss (default: 0.5). It currently supports 2D
inference only. Start with 4--8 steps and compare masks against the baseline;
TTA increases inference time by roughly three forward passes per step.
