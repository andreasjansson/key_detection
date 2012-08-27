Key Recognition with Zweiklang Profiles
=======================================

Andreas Jansson (School of Informatics, City University London)
Tillman Weyde (School of Informatics, City University London)


Command line format
-------------------

The executable is called `getkey`. To invoke it, provide -i and -o
command line flags as follows:

    ./getkey -i audio_file.wav -o output_file.tsv

The program uses a trained model that is stored in `model.pkl`. If you
invoke `getkey` from another directory than the directory in which
`getkey` resides, please point to the `model.pkl` file using the `-m`
argument, e.g.

    /some_directory/getkey -m /some_directory/model.pkl -i audio_file.wav -o output_file.tsv

For verbose output, invoke `getkey` with the `-v` flag, as follows:

    ./getkey -v -i audio_file.wav -o output_file.tsv


Requirements
------------

The program is written in Python, specifically Python 2.7. It requires
the `argparse`, `numpy` and `scipy` packages. It has been tested on
Linux, but will probably run on other platforms.


File structure
--------------

The program consists of two files: `getkey` is the main executable, a
self-contained Python module, and `model.pkl`, a "pickled" list of
Zweiklang profiles that has been trained prior to this submission.


Error handling
--------------

The main code block in `getkey` is wrapped in try..except, and if the
program fails during this block, it emits a warning and exists with
error code 1. If an input file results in all 0-amplitude spectrogram
(i.e., it is silent), a different warning is emitted and code 2 is
returned.
