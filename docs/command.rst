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
data, Cellpose can refine the predicted flow field on each test image without
labels. Candidate regions are connected components of the cell-probability map.
The objective makes endpoints compact *within each candidate region*, anchors
the result to the original flow, and penalizes locally discontinuous flow. The
checkpoint is never modified.

Enable it conservatively with a small number of steps::

    python -m cellpose --dir /path/to/images --tta_steps 8 --tta_lr 0.05

``--tta_region_threshold`` and ``--tta_min_region_size`` control which
cellprob components participate. ``--tta_max_points`` bounds the computation
per region. ``--tta_flow_weight`` and ``--tta_smooth_weight`` prevent the
compactness objective from excessively changing flow. It currently supports 2D
inference only. Start with 4--8 steps and compare masks against the baseline.
